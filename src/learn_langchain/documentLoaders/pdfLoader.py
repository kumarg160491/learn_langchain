import os
import warnings

from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain_community"
)

os.environ["USER_AGENT"] = "MyLangChainApp/1.0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader


class PDFDocumentReader:

    def __init__(self):
        self.model = ChatOllama(
            model="gemma2:2b-instruct-q5_0",
            temperature=0
        )

        self.file_path = (
            r"C:\Users\kumar\OneDrive\Documents\Learning"
            r"\Langchain\learn_langchain\Attention Is All You Need.pdf"
        )

        self.loader = PyMuPDF4LLMLoader(self.file_path)
        self.parser = StrOutputParser()

        self.promptcreation()

    def promptcreation(self):
        self.prompt = PromptTemplate(
            template="""
                    Write a comprehensive summary of the following PDF text.
                    
                    Focus on:
                    1. Main objective
                    2. Important concepts
                    3. Architecture
                    4. Key technical contributions
                    5. Important equations or mechanisms
                    6. Final conclusions
                    
                    PDF TEXT:
                    {pdf}
                    """,
            input_variables=["pdf"]
        )

    def pdf_loader(self):
        self.pages = self.loader.load()

    def get_page_content(self):
        self.full_text = "\n".join(
            page.page_content
            for page in self.pages
        )

    def chaincreation(self):
        chain = self.prompt | self.model | self.parser

        result = chain.invoke({
            "pdf": self.full_text
        })

        return result


if __name__ == "__main__":

    pdf_reader = PDFDocumentReader()

    pdf_reader.pdf_loader()

    print(f"Total pages: {len(pdf_reader.pages)}")

    pdf_reader.get_page_content()

    result = pdf_reader.chaincreation()

    print("\n===== PDF SUMMARY =====\n")
    print(result)
    with open("pdf_summary.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("\nSummary saved to pdf_summary.txt")