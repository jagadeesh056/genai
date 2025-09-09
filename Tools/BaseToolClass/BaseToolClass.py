from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(required="True", description="First Number to multiply")
    b: int = Field(required="True", description="Second Number to multiply")

class MultiplyTool(BaseTool):
    name: str = "Multiply"
    description: str = "Multiplication of two Numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b 
    
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a': 2, 'b': 9})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)

print(multiply_tool.args)

