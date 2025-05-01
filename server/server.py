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
    ES_INDEX = "business_school_updated"
    
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
    def build_graduate_queries(student_score: float, country: Optional[str] = None) -> Tuple[Dict, Dict]:
        logger.info(f"Building graduate queries with student_score: {student_score}, country: {country}")

        # Build country filter if needed
        country_filter = []
        if country and country != "None":
            country_filter.append({"term": {"location.keyword": country}})
            logger.debug(f"Added country filter: {country}")

        # Top 20 universities (score > student_score, smallest bigger ones first)
        top_query = {
            "size": 20,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"student__score": {"gt": student_score}}}
                    ] + country_filter
                }
            },
            "sort": [{"student__score": {"order": "asc"}}],
            "collapse": {
                "field": "business_school.keyword"
            }
        }

        # Bottom 10 universities (score < student_score, biggest smaller ones first)
        bottom_query = {
            "size": 10,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"student__score": {"lt": student_score}}}
                    ] + country_filter
                }
            },
            "sort": [{"student__score": {"order": "desc"}}],
            "collapse": {
                "field": "business_school.keyword"
            }
        }

        logger.debug(f"Top Query: {top_query}")
        logger.debug(f"Bottom Query: {bottom_query}")
        return top_query, bottom_query




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
    PREFERENCE_MAP = {
        1: 1.00,
        2: 1.33,
        3: 1.77,
        4: 2.36,
        5: 3.12
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
    
    @classmethod
    def preference_score(cls, request) -> dict:
        logger.info("Calculating preference scores from request parameters")

        rank_params = {
            'rank_university': request.GET.get('rank_university'),
            'rank_cost': request.GET.get('rank_cost'),
            'rank_employability': request.GET.get('rank_employability'),
            'rank_roi': request.GET.get('rank_roi'),
            'rank_workvisa': request.GET.get('rank_workvisa'),
        }

        # Step 1: Extract weights from ranks
        weights = {}
        for key, value in rank_params.items():
            try:
                rank = int(value)
                weights[key] = cls.PREFERENCE_MAP.get(rank, 0)
            except (ValueError, TypeError):
                weights[key] = 0

        total_weight = sum(weights.values())
        if total_weight == 0:
            logger.warning("Total weight is zero; assigning equal weights.")
            return {key: round(1 / len(weights), 2) for key in weights}

        # Step 2: Normalize and round to 2 decimals
        normalized = {
            key: round(value / total_weight, 2)
            for key, value in weights.items()
        }

        # Step 3: Adjust the last value to make total = 1.0
        keys = list(normalized.keys())
        total = sum(normalized.values())

        # if total != 1.0:
        #     diff = round(1.0 - total, 2)
        #     last_key = keys[-1]
        #     normalized[last_key] = round(normalized[last_key] + diff, 2)

        logger.info(f"Final normalized weights: {normalized}")
        return normalized

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
    print(request.GET.get("rank_employability", "None"))
    print(request.GET.get("rank_cost", "None"))
    try:
        student_score = StudentScorer.calculate_score(request)
        preference_score = StudentScorer.preference_score(request)
        country = request.GET.get("country", "None")
        logger.debug(f"Request parameters - student_score: {student_score}, country: {country}")
        logger.debug(f"Preference score: {preference_score}")

        top_query, bottom_query = ElasticsearchService.build_graduate_queries(student_score, country)

        top_response = ElasticsearchService.search(top_query)
        bottom_response = ElasticsearchService.search(bottom_query)

        top_hits = top_response["hits"]["hits"] if top_response and "hits" in top_response else []
        print("Top hits:", top_hits)
        bottom_hits = bottom_response["hits"]["hits"] if bottom_response and "hits" in bottom_response else []
        print("Bottom hits:", bottom_hits)

        logger.info(f"Top results received: {len(top_hits)}, Bottom results received: {len(bottom_hits)}")

        if len(top_hits) < 20:
            logger.warning(f"Expected 20 top universities but got only {len(top_hits)}.")
        if len(bottom_hits) < 10:
            logger.warning(f"Expected 10 bottom universities but got only {len(bottom_hits)}.")

        combined_hits = top_hits + bottom_hits

        # Extract data
        fees = [int(hit["_source"]["total_course_fees"]) for hit in combined_hits if hit["_source"].get("total_course_fees")]
        rankings = [int(hit["_source"]["study_bridge_ranking"]) for hit in combined_hits if hit["_source"].get("study_bridge_ranking")]
        roi_scores = [float(hit["_source"]["roi_score"]) for hit in combined_hits if hit["_source"].get("roi_score")]
        salaries = [int(hit["_source"]["salary"]) for hit in combined_hits if hit["_source"].get("salary")]
        employability = [float(hit["_source"]["employability"]) for hit in combined_hits if hit["_source"].get("employability")]
        work_visa = [int(hit["_source"]["work_visa_opportunities"]) for hit in combined_hits if hit["_source"].get("work_visa_opportunities")]

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
        for hit in combined_hits:
            source = hit["_source"]
            total_course_fees = int(source.get("total_course_fees", 0))
            ranking = int(source.get("study_bridge_ranking", 0))
            roi_score = float(source.get("roi_score", 0))
            salary = int(source.get("salary", 0))
            employability_score = float(source.get("employability", 0))
            work_visa_score = int(source.get("work_visa_opportunities", 0))

            weighage_percent_course_fee = round(average_fee / total_course_fees, 2) if average_fee else 0
            weighage_percent_ranking = round(average_ranking / ranking, 2) if average_ranking else 0
            weighage_percent_roi_score = round(roi_score / average_roi_score, 2) if average_roi_score else 0
            weighage_percent_salary = round(salary / average_salary, 2) if average_salary else 0
            weighage_percent_employability = round(employability_score / average_employability, 2) if average_employability else 0
            weighage_percent_work_visa = round(work_visa_score / average_work_visa, 2) if average_work_visa else 0

            aggregated_weighage_score = (
                (weighage_percent_course_fee * preference_score["rank_cost"])
                + (weighage_percent_ranking * preference_score["rank_university"])
                + (weighage_percent_roi_score * preference_score["rank_roi"])
                + (weighage_percent_employability * preference_score["rank_employability"])
                + (weighage_percent_work_visa * preference_score["rank_workvisa"])
            )

            logger.debug(f"Aggregated score for {source.get('business_school', 'N/A')}: {aggregated_weighage_score:.2f}")

            schools.append({
                "business_school": source.get("business_school", "N/A"),
                "university": source.get("university", "N/A"),
                "location": source.get("location", "N/A"),
                "course_duration_months": source.get("course_duration_(months)", "N/A"),
                "total_course_fees": total_course_fees,
                "student_score": source.get("student__score", "N/A"),
                "university_website": source.get("university_website", "#"),
                "aggregated_weighage_score": round(aggregated_weighage_score, 2)
            })

        # Sort and pick top 8 schools based on the aggregated_weighage_score
        top_schools = sorted(schools, key=lambda x: x["aggregated_weighage_score"], reverse=True)[:8]

        logger.info(f"Returning top {len(top_schools)} school recommendations based on aggregated score")
        return render(request, "recommendation.html", {"results": top_schools}) 

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