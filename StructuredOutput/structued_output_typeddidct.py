from pydantic import BaseModel, Field
from typing_extensions import Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

# Use ChatGoogleGenerativeAI directly
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

class Review(BaseModel):
    key_themes: list[str] = Field(description = "Write down all the key themes discussed in the review in a list")
    summary: str = Field(description = "Write the brief summary of review")
    sentiment: Literal["pos", "neg","neutral"] = Field(description = "Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="what are the pros of review")
    cons: Optional[list[str]] = Field(default=None, description="what are the cons of review")
    name: Optional[str] = Field(default=None, description="write the name of reviewer")

structured_output = model.with_structured_output(Review)
result = structured_output.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
reviewed by reddy""")
print(result)