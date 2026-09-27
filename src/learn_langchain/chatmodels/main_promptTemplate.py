from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question: {question}",
    validate_template=True,)

prompt.save("prompt_template.json")

