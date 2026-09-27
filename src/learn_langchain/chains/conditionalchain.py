import os
from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN"
    ),
    temperature=0,
    max_new_tokens=1500,)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Gives the sentiment as positive or negative")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write the appropriate response for the negative feedback \n {feedback}',
    input_variables=['feedback'],
)


prompt3 = PromptTemplate(
    template='Write the appropriate response for the positive feedback \n {feedback}',
    input_variables=['feedback'],
)

classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'Could not find sentiment')

)

chain = classifier_chain | branch_chain
print(chain.invoke({'feedback': 'This is a terrible smartphone'}))
chain.get_graph().print_ascii()