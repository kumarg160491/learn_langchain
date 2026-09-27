from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.9)
result =llm.invoke("Write a poem about the beauty of nature.")
print(result.content)