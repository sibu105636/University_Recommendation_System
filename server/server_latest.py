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

# Fetch Data from Elasticsearch
def fetch_data_from_elasticsearch():
    url = f"{ES_HOST}/{ES_INDEX}/_search"
    headers = {"Content-Type": "application/json"}
    query = {"size": 10000, "query": {"match_all": {}}}
    response = requests.get(url, headers=headers, json=query)

    if response.status_code == 200:
        records = [hit["_source"] for hit in response.json()["hits"]["hits"]]
        return pd.DataFrame(records)
    else:
        print("Error fetching data:", response.json())  # Debugging info
        return pd.DataFrame()

# Convert request values safely
def get_float(value, default=0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Helper functions replacing `student_input` imports
def year_of_exp(value): return value * 1.0
def designation(value): return category_score(value)
def company_type(value): return category_score(value)
def annual_scale_of_business(value): return category_score(value)
def ngo(value): return category_score(value)
def gmat_score(value): return value
def cgpa_score(value): return value
def ielts(value): return value
def college_ranking(value): return category_score(value)
def course_relevance(value): return category_score(value)

# Create student score from form data
def create_student_score(request):
    work_ex = get_float(request.POST.get("workEx", 0))
    desg = request.POST.get("Desg", "")
    com_ty = request.POST.get("comTy", "")
    ann_scale = request.POST.get("annScale", "")
    ngo_value = request.POST.get("ngo", "")
    gmat = get_float(request.POST.get("gmat", 0))
    cgpa = get_float(request.POST.get("cgpa", 0))
    ielts_score = get_float(request.POST.get("ielts", 0))
    col_rnk = request.POST.get("colRnk", "")
    co_re = request.POST.get("coRe", "")

    values = [
        year_of_exp(work_ex) * 0.1006,  
        designation(desg) * 0.0442,  
        company_type(com_ty) * 0.0748,
        annual_scale_of_business(ann_scale) * 0.1535,  
        ngo(ngo_value) * 0.04,  
        gmat_score(gmat) * 0.2853,  
        cgpa_score(cgpa) * 0.1219,  
        ielts(ielts_score) * 0.0268,  
        college_ranking(col_rnk) * 0.0979,  
        course_relevance(co_re) * 0.0551
    ]
    return sum(values)

# KNN Algorithm
def euclidean_dist(instance1, instance2):
    return np.linalg.norm(instance1 - instance2)

def knn(trainSet, test_instance, k):
    test_values = test_instance.values.flatten()

    distances = [
        (x, euclidean_dist(test_values, trainSet.iloc[x][:-1].values))  
        for x in range(len(trainSet))
    ]

    sorted_distances = sorted(distances, key=lambda x: x[1])
    neighbors_list = [sorted_distances[i][0] for i in range(min(k, len(trainSet)))]

    class_votes = {}
    for idx in neighbors_list:
        label = trainSet.iloc[idx, -1]  # Assuming last column is the label
        class_votes[label] = class_votes.get(label, 0) + 1

    sorted_neighbors = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_neighbors[0][0] if sorted_neighbors else "Unknown", neighbors_list

# Undergraduate Recommendation Logic
def undergraduatealgo(request):
    result = [("University A", 90), ("University B", 85), ("University C", 80), ("University D", 75), ("University E", 70)]
    return render(request, 'recommendation.html', {'results': result})

def graduatealgo(request):
    if request.method == "POST":
        student_score = create_student_score(request)

        query = {
            "size": 10,
            "query": {
                "bool": {
                    "must": {
                        "script_score": {
                            "query": { "match_all": {} },
                            "script": {
                                "source": "doc.containsKey(\"student__score\") && doc[\"student__score\"].size() > 0 ? 1 / (1 + Math.abs(params.student_score - doc[\"student__score\"].value)) : 0",
                                "params": { "student_score": float(student_score) }
                            }
                        }
                    },
                    "filter": {
                        "range": {
                            "student__score": {
                                "gte": float(student_score) - 0.1,
                                "lte": float(student_score) + 0.1
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

        if response.status_code == 200:
            results = response.json()["hits"]["hits"]
            schools = [(hit["_source"]["business_school"], round(hit["_score"] * 100, 2)) for hit in results]
        else:
            return HttpResponse(f"Error fetching data: {response.json()}", status=500)

        return render(request, 'recommendation.html', {'results': schools})
    else:
        return HttpResponse("Invalid request method", status=405)

# URL Patterns
urlpatterns = [
    path('', index, name='index'),
    path('graduate', graduate, name='graduate'),
    path('undergraduate', undergraduate, name='undergraduate'),
    path('undergraduatealgo', undergraduatealgo, name='undergraduatealgo'),
    path('graduatealgo', graduatealgo, name='graduatealgo'),
]

# Run Django Server
if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line(["manage.py", "runserver", "0.0.0.0:8000"])
