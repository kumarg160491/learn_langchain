import os
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv(
#         "HUGGINGFACEHUB_ACCESS_TOKEN"
#     ),
#     temperature=0,
#     max_new_tokens=1500,
# )
#
# model = ChatHuggingFace(llm=llm)
model = ChatOllama(
    model="gemma2:2b-instruct-q5_0",
    temperature=0,
)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_generation_chain = RunnableSequence(
    prompt1, model, parser
)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>200, RunnableSequence(prompt2, model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_generation_chain, branch_chain
)

result = final_chain.invoke({'topic': 'Russia Vs Ukraine'})
print(result)