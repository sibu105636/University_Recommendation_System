def year_of_exp(y):
    if 4.5 < y < 6:
        return 5
    elif (3.5 <= y <= 4.5) or (6 <= y <= 7):
        return 4
    elif (2.5 <= y <= 3.5) or (7 <= y <= 9):
        return 3
    elif (1.5 <= y <= 2.5) or (9 <= y <= 12):
        return 2
    elif (0 <= y <= 1.5) or (12 <= y <= 15):
        return 1
    else:
        return 0
    
import re

def designation(d4):
    category_mapping = {
        "Category1": 5,
        "Category 2": 4,
        "Category 3": 3,
        "Category 4": 2,
        "Category 5": 1
    }
    
    d4 = str(d4).strip()  # Convert to string and remove leading/trailing spaces
    d4 = re.sub(r'\s+', ' ', d4)  # Normalize spaces

    return category_mapping.get(d4, 0)  #

def company_type(d5):
    category_mapping = {
        "Category 1": 5,
        "Category 2": 4,
        "Category 3": 3,
        "Category 4": 2,
        "Category 5": 1
    }
    d4 = str(d5).strip()  # ✅ Convert to string first
    return category_mapping.get(d5, 0) 

def annual_scale_of_business(d6):
    category_mapping = {
        "Category 1": 5,
        "Category 2": 4,
        "Category 3": 3,
        "Category 4": 2,
        "Category 5": 1
    }
    d4 = str(d6).strip()  # ✅ Convert to string first
    return category_mapping.get(d6, 0) 

def ngo(d_value):
    category_mapping = {
        "Category 1": 5,
        "Category 2": 4,
        "Category 3": 3,
        "Category 4": 2,
        "Category 5": 1
    }
    d4 = str(d_value).strip()  # ✅ Convert to string first
    return category_mapping.get(d_value, 0) 

def gmat_score(d8):
    if d8 >= 800:
        return 5
    elif d8 >= 720:
        return 4 + 0.0125 * (d8 - 720)
    elif d8 >= 660:
        return 3 + 0.0167 * (d8 - 660)
    elif d8 >= 600:
        return 2 + 0.0167 * (d8 - 600)
    elif d8 >= 550:
        return 1 + 0.02 * (d8 - 550)
    else:
        return 0
    
def cgpa_score(d9):
    if d9 >= 4.2:
        return 5
    elif d9 >= 4:
        return 4.5 + (d9 - 4) * 2.5
    elif d9 >= 3.5:
        return 3.5 + (d9 - 3.5) * 2
    elif d9 >= 3:
        return 2.5 + (d9 - 3) * 2
    elif d9 >= 2.5:
        return 1.5 + (d9 - 2.5) * 2
    elif d9 >= 1.5:
        return (d9 - 1.5) * 1
    else:
        return 0

def ielts(d10):
    if d10 >= 9:
        return 5
    elif d10 >= 8.5:
        return 4.5 + (d10 - 8.5) * 2
    elif d10 >= 8:
        return 4 + (d10 - 8) * 1
    elif d10 >= 7.5:
        return 3.5 + (d10 - 7.5) * 1
    elif d10 >= 7:
        return 3 + (d10 - 7) * 1
    elif d10 >= 6.5:
        return 2.5 + (d10 - 6.5) * 1
    elif d10 >= 6:
        return 2 + (d10 - 6) * 1
    elif d10 >= 5.5:
        return 1.5 + (d10 - 5.5) * 1
    elif d10 >= 5:
        return 1 + (d10 - 5) * 1
    else:
        return 0
    
def college_ranking(d11):
    category_mapping = {
        "Category 1": 5,
        "Category 2": 4,
        "Category 3": 3,
        "Category 4": 2,
        "Category 5": 1
    }
    d4 = str(d11).strip()  # ✅ Convert to string first
    return category_mapping.get(d11, 0) 

def course_relevance(d12):
    category_mapping = {
        "Category 1": 5,
        "Category 2": 4,
        "Category 3": 3,
        "Category 4": 2,
        "Category 5": 1
    }
    d4 = str(d12).strip()  # ✅ Convert to string first
    return category_mapping.get(d12, 0) 