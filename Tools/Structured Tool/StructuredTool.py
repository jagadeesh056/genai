from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class Multiply_Input(BaseModel):
    a: int = Field(required="True", description="First Number to add")
    b: int = Field(required="True", description="Second Number to add")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="Multiply",
    description="Multiply two numbers",
    args_schema=Multiply_Input
)

result = multiply_tool.invoke({"a": 3, "b": 7})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)