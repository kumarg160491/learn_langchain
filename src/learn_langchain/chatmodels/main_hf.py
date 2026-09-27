import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.load import loads
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from warnings import filterwarnings

load_dotenv()

llm = HuggingFaceEndpoint(
		repo_id="openai/gpt-oss-120b",
	task="text-generation",
		huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
	temperature=0.7,
	max_new_tokens=1500,
)

# with open("prompt_template.json", "r", encoding="utf-8") as f:
#     template = loads(f.read())
template = load_prompt('prompt_template.json')

model = ChatHuggingFace(llm=llm)

#LCEL pattern
chain = template | model | StrOutputParser()
question = 'What is the capital of India?'

response = chain.invoke({
    'question': question
})

print(response)