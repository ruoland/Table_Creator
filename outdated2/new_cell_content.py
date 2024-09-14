import random
import string
from datetime import datetime, timedelta
from new_constants import *
def generate_cell_content():
    content_type = random.choice(['text', 'number', 'date', 'time', 'name', 'subject', 'room', 'grade'])
    
    if content_type == 'text':
        return random.choice(COMMON_WORDS)
    
    elif content_type == 'number':
        return str(random.randint(0, 1000))
    
    elif content_type == 'date':
        start_date = datetime(2023, 1, 1)
        random_date = start_date + timedelta(days=random.randint(0, 365))
        return random_date.strftime("%Y-%m-%d")
    
    elif content_type == 'time':
        return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"
    
    elif content_type == 'name':
        return f"{random.choice(PROFESSORS)} 교수"
    
    elif content_type == 'subject':
        return random.choice(SUBJECTS)
    
    elif content_type == 'room':
        return f"{random.choice(BUILDINGS)} {random.randint(100, 500)}호"
    
    elif content_type == 'grade':
        return random.choice(GRADES)
    
    else:
        return "N/A"

def generate_student_id():
    year = random.randint(2015, 2023)
    number = random.randint(1000, 9999)
    return f"{year}{number:04d}"

def generate_course_code():
    department = ''.join(random.choices(string.ascii_uppercase, k=3))
    number = random.randint(100, 499)
    return f"{department}{number}"