from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-4", temperature=0.7, max_tokens=1500, max_completion_tokens=1500)
result = model.invoke("Hello! How can I assist you today?")
print(result.content)