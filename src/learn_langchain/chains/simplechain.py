import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN"
    ),
    temperature=0,
    max_new_tokens=1500,
)

prompt = PromptTemplate(
    template='Generate 5 interesting facts on {topic}',
    input_variables=['topic'],
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({'topic': 'cricket'})
# print(response)

chain.get_graph().print_ascii()