import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN
    ),
    temperature=0,
    max_new_tokens=1500,
)

class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(description="Age of person", gt=18)
    city: str = Field(description="City of person")

parser = PydanticOutputParser(pydantic_object=Person)

tempalte = PromptTemplate(
    template='Generate the name , age and city of a frictional {place} person \n {format_instruction} '
)