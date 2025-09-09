from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """multiplies two numbers"""
    return a * b 

result = multiply.invoke({"a": 3,"b": 20})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())