from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal

class Student(BaseModel):
    
    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: Optional[float] = Field(gt=0, lt=10, default=5.0,  description="CGPA will be between 0 to 10")

new_student = {'age': 24, 'email': 'abc@gmail.com' }
student = Student(**new_student)

student_dict = student.model_dump(include = {'name', 'age', 'abc'})
student_json = student.model_dump_json()

print(student_dict)
print(student_json)