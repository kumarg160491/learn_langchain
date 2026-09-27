import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.load import loads
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from warnings import filterwarnings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
		repo_id="openai/gpt-oss-120b",
	task="text-generation",
		huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
	temperature=0.7,
	max_new_tokens=1500,
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpfu assistant"),
    HumanMessage(content="Tell me about langchain and langgraph")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)

