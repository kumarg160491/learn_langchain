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
	temperature=0,
	max_new_tokens=1500,
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

template1 = PromptTemplate(
    template = 'Give me the detailed report on {topic}',
    input_variable = ['topic']
)

template2 = PromptTemplate(
    template='write the 5 line summary on the following text. \n {text}',
    input_variables=['text']
)
chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({'topic': 'black hole'})
print(response)