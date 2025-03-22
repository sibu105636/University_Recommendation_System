import random
import requests
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from django.urls import path
from django.conf import settings
from django.core.wsgi import get_wsgi_application
import os
import string
from student_input import year_of_exp, designation, company_type, annual_scale_of_business, ngo, gmat_score, cgpa_score, ielts, college_ranking, course_relevance

# Elasticsearch Configuration
ES_HOST = "http://127.0.0.1:9200"
ES_INDEX = "business_schools"

# Generate a random SECRET_KEY if not set
def generate_secret_key():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=50))

# Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
settings.configure(
    DEBUG=True,
    SECRET_KEY=os.getenv("DJANGO_SECRET_KEY", generate_secret_key()),
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=['*'],
    TEMPLATES=[{'BACKEND': 'django.template.backends.django.DjangoTemplates', 'DIRS': ['templates'], 'APP_DIRS': True}],
)
application = get_wsgi_application()

# Views
def index(request):
    return render(request, 'index.html')

def graduate(request):
    return render(request, 'graduate.html')

def undergraduate(request):
    return render(request, 'undergraduate.html')

# Function to assign scores for categorical variables
def category_score(value):
    category_mapping = {
        "low": 1, "medium": 2, "high": 3, 
        "junior": 1, "mid": 2, "senior": 3
    }
    return category_mapping.get(value.lower(), 0) if isinstance(value, str) else 0

# Helper function to fetch data from Elasticsearch
def fetch_data_from_elasticsearch():
    url = f"{ES_HOST}/{ES_INDEX}/_search"
    headers = {"Content-Type": "application/json"}
    query = {"size": 10000, "query": {"match_all": {}}}
    response = requests.get(url, headers=headers, json=query)

    if response.status_code == 200:
        records = [hit["_source"] for hit in response.json()["hits"]["hits"]]
        return pd.DataFrame(records)
    else:
        print("Error fetching data:", response.json())  # Log error for debugging
        return pd.DataFrame()

# Function to safely convert request values to float
def get_float(value, default=0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Function to handle categorical data
def get_category(value):
    return category_score(value) if value else 0  # Ensure 0 is returned for missing values

# Function to create a DataFrame from request parameters
def create_student_score(request):
    print(designation(request.GET.get("Desg", "")))
    print(year_of_exp(get_float(request.GET.get("workEx", 0))))
    print(company_type(request.GET.get("comTy", "")))
    print(annual_scale_of_business(request.GET.get("annScale", "")))
    print(ngo(request.GET.get("ngo", "")))
    print(gmat_score(get_float(request.GET.get("gmat", 0))))
    print(cgpa_score(get_float(request.GET.get("cgpa", 0))))
    print(ielts(get_float(request.GET.get("ielts", 0))) )
    print(college_ranking(request.GET.get("colRnk", "")))
    print(request.GET.get("colRnk", ""))
    print(course_relevance(request.GET.get("coRe", "")))


    
    values = [
        year_of_exp(get_float(request.GET.get("workEx", 0))) * 0.1006,  
        designation(request.GET.get("Desg", "")) * 0.0442,  
        company_type(request.GET.get("comTy", ""))* 0.0748,
        annual_scale_of_business(request.GET.get("annScale", "")) * 0.1535,  
        ngo(request.GET.get("ngo", "")) * 0.04,  
        gmat_score(get_float(request.GET.get("gmat", 0))) * 0.2853,  
        cgpa_score(get_float(request.GET.get("cgpa", 0))) * 0.1219,  
        ielts(get_float(request.GET.get("ielts", 0))) * 0.0268,  
        college_ranking(request.GET.get("colRnk", "")) * 0.0979,  
        course_relevance(request.GET.get("coRe", "")) * 0.0551
    ]
    return sum(values)
    # return pd.DataFrame([values], columns=keys)

# KNN Algorithm
def euclidean_dist(instance1, instance2):
    return np.linalg.norm(instance1 - instance2)

def knn(trainSet, test_instance, k):
    test_values = test_instance.values.flatten()  # Ensure test instance is a 1D array

    distances = [
        (x, euclidean_dist(test_values, trainSet.iloc[x][:-1].values))  
        for x in range(len(trainSet))
    ]

    sorted_distances = sorted(distances, key=lambda x: x[1])
    neighbors_list = [sorted_distances[i][0] for i in range(min(k, len(trainSet)))]  # Handle small datasets

    class_votes = {}
    for idx in neighbors_list:
        label = trainSet.iloc[idx, -1]  # Assuming last column is the label
        class_votes[label] = class_votes.get(label, 0) + 1

    sorted_neighbors = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_neighbors[0][0] if sorted_neighbors else "Unknown", neighbors_list

# Undergraduate Recommendation Logic
def undergraduatealgo(request):
    # TODO: Replace with real logic
    result = [("University A", 90), ("University B", 85), ("University C", 80), ("University D", 75), ("University E", 70)]
    return render(request, 'recommendation.html', {'results': result})

def graduatealgo(request):
    test = create_student_score(request)  # Calculate student's score

    query = {
        "size": 10,
        "query": {
            "bool": {
                "must": {
                    "script_score": {
                        "query": { "match_all": {} },
                        "script": {
                            "source": "doc.containsKey(\"student__score\") && doc[\"student__score\"].size() > 0 ? 1 / (1 + Math.abs(params.student_score - doc[\"student__score\"].value)) : 0",
                            "params": { "student_score": float(test) }
                        }
                    }
                },
                "filter": {
                    "range": {
                        "student__score": {
                            "gte": float(test) - 0.1,
                            "lte": float(test) + 0.1
                        }
                    }
                }
            }
        },
        "sort": [{"_score": "desc"}],
        "collapse": {
            "field": "business_school.keyword"
        }
    }

    response = requests.get(f"{ES_HOST}/{ES_INDEX}/_search", json=query)
    print(response.json())

    if response.status_code == 200:
        results = response.json()["hits"]["hits"]

        schools = [
            {
                "business_school": hit["_source"].get("business_school", "N/A"),
                "university": hit["_source"].get("university", "N/A"),
                "location": hit["_source"].get("location", "N/A"),
                "course_duration_months": hit["_source"].get("course_duration_(months)", "N/A"),  # Renamed key
                "total_course_fees": hit["_source"].get("total_course_fees", "N/A"),
                "university_website": hit["_source"].get("university_website", "#"),  # Add website if available
            }
            for hit in results
        ]
    else:
        return HttpResponse(f"Error fetching data: {response.json()}", status=500)

    return render(request, 'recommendation.html', {'results': schools})




# URL Patterns
urlpatterns = [
    path('', index, name='index'),
    path('main', index, name='index'),
    path('graduate', graduate, name='graduate'),
    path('undergraduate', undergraduate, name='undergraduate'),
    path('undergraduatealgo', undergraduatealgo, name='undergraduatealgo'),
    path('graduatealgo', graduatealgo, name='graduatealgo'),
]

# Django runserver command
if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line(["manage.py", "runserver", "0.0.0.0:8000"])
