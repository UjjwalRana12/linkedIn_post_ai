from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
app = FastAPI()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=api_key)
class PostRequest(BaseModel): 
    article_summary: str
    viewpoints: list[str]
class PostResponse(BaseModel): 
    linkedin_post: str = Field(
        description="""Write an engaging LinkedIn post (200-250 words) from the perspective of a physician passionate about healthcare AI. Reference the article summary and incorporate all viewpoints provided.
        Include a hook, specific insights, and a clear call-to-action. Use professional tone without hashtags or emojis.""")
    Confidence_Score: float = Field(gt=0, lt=1,  description="Provide a confidence score between 0 and 1 representing how well the post integrates all the provided viewpoints" )

structured_model = llm.with_structured_output(PostResponse)
@app.post("/generate_linkedin_post", response_model=PostResponse)
async def generate_linkedin_post(request: PostRequest):
    prompt = f"Article Summary: {request.article_summary}\n\nViewpoints:\n" + "\n".join(request.viewpoints)
    result = structured_model.invoke(prompt)  
    return result