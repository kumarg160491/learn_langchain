import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN"
    ),
    temperature=0,
    max_new_tokens=1500,
)

model = ChatHuggingFace(llm=llm)

# JSON Parser
parser = JsonOutputParser()

# Prompt 1: Generate detailed report
report_prompt = PromptTemplate(
    template="""
    Give me a detailed report on {topic}.
    """,
    input_variables=["topic"],
)

# Prompt 2: Generate summary in JSON format
summary_prompt = PromptTemplate(
    template="""
    Summarize the following report in 5 lines.

    Report:
    {report}

    {format_instructions}
    """,
    input_variables=["report"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# LCEL Chain
chain = (
    report_prompt
    | model
    | RunnableLambda(
        lambda msg: {"report": msg.content}
    )
    | summary_prompt
    | model
    | parser
)

# Execute
response = chain.invoke({
    "topic": "black hole"
})

print(response)
print('***************************')
print(response['summary'])