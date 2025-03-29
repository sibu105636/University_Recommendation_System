import random
import string
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from django.shortcuts import render, HttpResponse
from django.urls import path
from django.conf import settings
from django.core.wsgi import get_wsgi_application
import requests
import numpy as np
import pandas as pd
from student_input import (
    year_of_exp, designation, company_type, 
    annual_scale_of_business, ngo, gmat_score, 
    cgpa_score, ielts, college_ranking, course_relevance
)

# --------------------------
# Logger Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Configuration Module
# --------------------------
class Config:
    ES_HOST = "http://127.0.0.1:9200"
    ES_INDEX = "business_schools"
    
    @staticmethod
    def generate_secret_key():
        return ''.join(random.choices(string.ascii_letters + string.digits, k=50))
    
    @staticmethod
    def configure_django():
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
        settings.configure(
            DEBUG=True,
            SECRET_KEY=os.getenv("DJANGO_SECRET_KEY", Config.generate_secret_key()),
            ROOT_URLCONF=__name__,
            ALLOWED_HOSTS=['*'],
            TEMPLATES=[{
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'DIRS': ['templates'],
                'APP_DIRS': True
            }],
        )

# --------------------------
# Elasticsearch Service Module
# --------------------------
class ElasticsearchService:
    @staticmethod
    def build_graduate_query(student_score: float, country: Optional[str] = None) -> Dict:
        logger.info(f"Building graduate query with student_score: {student_score}, country: {country}")
        query = {
            "size": 30,  # Return top 30 closest neighbors
            "query": {
                "bool": {
                    "filter": [],  # Explicitly define filter list to prevent KeyError
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": """
                                        if (doc.containsKey('student__score') && doc['student__score'].size() > 0) {
                                            return 1 / (1 + Math.abs(params.student_score - doc['student__score'].value));
                                        } else {
                                            return 0;
                                        }
                                    """,
                                    "params": {"student_score": float(student_score)}
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"_score": "desc"}],  # Higher score = closer neighbor
            "collapse": {
                "field": "business_school.keyword"  # Ensures only one document per business school
            }
        }

        # Apply country filter if provided
        if country and country != "None":
            query["query"]["bool"]["filter"].append({"term": {"location.keyword": country}})
            logger.debug(f"Added country filter for: {country}")

        logger.debug(f"Final query: {query}")
        return query

    @staticmethod
    def search(query: Dict) -> Dict:
        url = f"{Config.ES_HOST}/{Config.ES_INDEX}/_search"
        headers = {"Content-Type": "application/json"}
        logger.info(f"Making Elasticsearch request to: {url}")
        
        try:
            response = requests.get(url, headers=headers, json=query)
            if response.status_code == 200:
                logger.info("Elasticsearch request successful")
                return response.json()
            else:
                logger.error(f"Elasticsearch request failed with status {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Elasticsearch connection error: {str(e)}")
            return None

# --------------------------
# Student Scoring Module
# --------------------------
class StudentScorer:
    WEIGHTS = {
        'work_experience': 0.1006,
        'designation': 0.0442,
        'company_type': 0.0748,
        'business_scale': 0.1535,
        'ngo': 0.04,
        'gmat': 0.2853,
        'cgpa': 0.1219,
        'ielts': 0.0268,
        'college_ranking': 0.0979,
        'course_relevance': 0.0551
    }

    @staticmethod
    def get_float(value, default=0) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert value '{value}' to float, using default {default}")
            return default

    @classmethod
    def calculate_score(cls, request) -> float:
        logger.info("Calculating student score from request parameters")
        params = {
            'work_experience': year_of_exp(cls.get_float(request.GET.get("workEx", 0))),
            'designation': designation(request.GET.get("Desg", "")),
            'company_type': company_type(request.GET.get("comTy", "")),
            'business_scale': annual_scale_of_business(request.GET.get("annScale", "")),
            'ngo': ngo(request.GET.get("ngo", "")),
            'gmat': gmat_score(cls.get_float(request.GET.get("gmat", 0))),
            'cgpa': cgpa_score(cls.get_float(request.GET.get("cgpa", 0))),
            'ielts': ielts(cls.get_float(request.GET.get("ielts", 0))),
            'college_ranking': college_ranking(request.GET.get("colRnk", "")),
            'course_relevance': course_relevance(request.GET.get("coRe", ""))
        }

        logger.debug(f"Score parameters: {params}")
        score = sum(value * cls.WEIGHTS[key] for key, value in params.items())
        logger.info(f"Calculated student score: {score}")
        return score

# --------------------------
# KNN Algorithm Module
# --------------------------
class KNNService:
    @staticmethod
    def euclidean_dist(instance1, instance2) -> float:
        return np.linalg.norm(instance1 - instance2)

    @classmethod
    def predict(cls, train_set: pd.DataFrame, test_instance: pd.DataFrame, k: int = 5) -> Tuple[str, List]:
        logger.info(f"Running KNN prediction with k={k}")
        test_values = test_instance.values.flatten()
        
        distances = [
            (idx, cls.euclidean_dist(test_values, train_set.iloc[idx][:-1].values))
            for idx in range(len(train_set))
        ]
        
        sorted_distances = sorted(distances, key=lambda x: x[1])
        neighbors = [sorted_distances[i][0] for i in range(min(k, len(train_set)))]
        
        class_votes = {}
        for idx in neighbors:
            label = train_set.iloc[idx, -1]
            class_votes[label] = class_votes.get(label, 0) + 1
        
        sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
        result = (sorted_votes[0][0] if sorted_votes else "Unknown", neighbors)
        logger.info(f"KNN prediction result: {result[0]}")
        return result

# --------------------------
# View Controllers
# --------------------------
def index(request):
    logger.info("Handling index page request")
    return render(request, 'index.html')

def graduate(request):
    logger.info("Handling graduate page request")
    return render(request, 'graduate.html')

def undergraduate(request):
    logger.info("Handling undergraduate page request")
    return render(request, 'undergraduate.html')

def undergraduatealgo(request):
    logger.info("Handling undergraduate algorithm request")
    # TODO: Replace with real logic
    result = [("University A", 90), ("University B", 85), ("University C", 80)]
    logger.debug(f"Undergraduate results: {result}")
    return render(request, 'recommendation.html', {'results': result})

def graduatealgo(request):
    logger.info("Handling graduate algorithm request")
    try:
        student_score = StudentScorer.calculate_score(request)
        country = request.GET.get("country", "None")
        logger.debug(f"Request parameters - student_score: {student_score}, country: {country}")

        query = ElasticsearchService.build_graduate_query(student_score, country)
        response = ElasticsearchService.search(query)

        if not response:
            logger.error("No response from Elasticsearch")
            return HttpResponse("Error fetching data from Elasticsearch", status=500)

        # Extracting values
        fees = [
            int(hit["_source"]["total_course_fees"])
            for hit in response["hits"]["hits"]
            if hit["_source"].get("total_course_fees")
        ]
        rankings = [
            int(hit["_source"]["study_bridge_ranking"])
            for hit in response["hits"]["hits"]
            if hit["_source"].get("study_bridge_ranking")
        ]
        roi_scores = [
            float(hit["_source"]["roi_score"])
            for hit in response["hits"]["hits"]
            if hit["_source"].get("roi_score")
        ]
        salaries = [
            int(hit["_source"]["salary"])
            for hit in response["hits"]["hits"]
            if hit["_source"].get("salary")
        ]
        employability = [
            float(hit["_source"]["employability"])
            for hit in response["hits"]["hits"]
            if hit["_source"].get("employability")
        ]
        work_visa = [
            int(hit["_source"]["work_visa_opportunities"])
            for hit in response["hits"]["hits"]
            if hit["_source"].get("work_visa_opportunities")
        ]

        # Calculating averages
        average_fee = sum(fees) / len(fees) if fees else 0
        average_ranking = sum(rankings) / len(rankings) if rankings else 0
        average_roi_score = sum(roi_scores) / len(roi_scores) if roi_scores else 0
        average_salary = sum(salaries) / len(salaries) if salaries else 0
        average_employability = sum(employability) / len(employability) if employability else 0
        average_work_visa = sum(work_visa) / len(work_visa) if work_visa else 0

        logger.debug(f"Calculated averages - Fee: {average_fee:.2f}, Ranking: {average_ranking:.2f}, "
                    f"ROI: {average_roi_score:.2f}, Salary: {average_salary:.2f}, "
                    f"Employability: {average_employability:.2f}, Work Visa: {average_work_visa:.2f}")

        schools = []
        for hit in response["hits"]["hits"]:
            total_course_fees = int(hit["_source"].get("total_course_fees", 0))
            ranking = int(hit["_source"].get("study_bridge_ranking", 0))
            roi_score = float(hit["_source"].get("roi_score", 0))
            salary = int(hit["_source"].get("salary", 0))
            employability_score = float(hit["_source"].get("employability", 0))
            work_visa_score = int(hit["_source"].get("work_visa_opportunities", 0))

            # Calculate weightage percentages
            weighage_percent_course_fee = (
                round(average_fee / total_course_fees, 2) if average_fee else 0
            )
            weighage_percent_ranking = (
                round(average_ranking / ranking, 2) if average_ranking else 0
            )
            weighage_percent_roi_score = (
                round(roi_score / average_roi_score, 2) if average_roi_score else 0
            )
            weighage_percent_salary = (
                round(salary / average_salary, 2) if average_salary else 0
            )
            weighage_percent_employability = (
                round(employability_score / average_employability, 2)
                if average_employability
                else 0
            )
            weighage_percent_work_visa = (
                round(work_visa_score / average_work_visa, 2) if average_work_visa else 0
            )

            school_data = {
                "business_school": hit["_source"].get("business_school", "N/A"),
                "university": hit["_source"].get("university", "N/A"),
                "location": hit["_source"].get("location", "N/A"),
                "course_duration_months": hit["_source"].get(
                    "course_duration_(months)", "N/A"
                ),
                "total_course_fees": total_course_fees,
                "weighage_percent_course_fee": weighage_percent_course_fee,
                "weighage_percent_ranking": weighage_percent_ranking,
                "weighage_percent_roi_score": weighage_percent_roi_score,
                "weighage_percent_salary": weighage_percent_salary,
                "weighage_percent_employability": weighage_percent_employability,
                "weighage_percent_work_visa": weighage_percent_work_visa,
                "student_score": hit["_source"].get("student__score", "N/A"),
                "university_website": hit["_source"].get("university_website", "#"),
            }
            schools.append(school_data)
            logger.debug(f"Processed school data: {school_data}")

        logger.info(f"Returning {len(schools)} school recommendations")
        return render(request, "recommendation.html", {"results": schools})
    except Exception as e:
        logger.error(f"Error in graduatealgo: {str(e)}", exc_info=True)
        return HttpResponse("An error occurred while processing your request", status=500)

# --------------------------
# Application Setup
# --------------------------
urlpatterns = [
    path('', index, name='index'),
    path('main', index, name='index'),
    path('graduate', graduate, name='graduate'),
    path('undergraduate', undergraduate, name='undergraduate'),
    path('undergraduatealgo', undergraduatealgo, name='undergraduatealgo'),
    path('graduatealgo', graduatealgo, name='graduatealgo'),
]

if __name__ == "__main__":
    logger.info("Starting application")
    Config.configure_django()
    application = get_wsgi_application()
    from django.core.management import execute_from_command_line
    execute_from_command_line(["manage.py", "runserver", "0.0.0.0:8000"])