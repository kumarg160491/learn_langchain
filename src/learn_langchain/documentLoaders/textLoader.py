import os
from dotenv import load_dotenv

# Set the USER_AGENT before LangChain loads to silence the web loader warning
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # Modern replacement
from langchain_core.prompts import PromptTemplate

# 1. Initialize your local Ollama model
model = ChatOllama(
    model="gemma2:2b-instruct-q5_0",
    temperature=0
)

# 2. Define the Prompt and Parser
prompt = PromptTemplate(
    template="Write a summary for the following poem:\n\n{poem}",
    input_variables=['poem']
)
parser = StrOutputParser()

# 3. Modern replacement for TextLoader using standard Python file reading
# This reads the file safely and packages it into a LangChain Document structure
file_path = 'cricket.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    text_content = f.read()

# Create a LangChain Document object matching the exact structure your chain expects
doc = [Document(page_content=text_content, metadata={"source": file_path})]

# 4. Construct and invoke the LCEL chain
chain = prompt | model | parser

result = chain.invoke({'poem': doc[0].page_content})

# Print out your summary
print(result)
