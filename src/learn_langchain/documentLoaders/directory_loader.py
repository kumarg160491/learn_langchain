from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)


pdf_loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyMuPDF4LLMLoader,
)

text_loader = DirectoryLoader(
    path="books",
    glob="*.txt",
    loader_cls=TextLoader,
)

pdf_documents = pdf_loader.lazy_load()
text_documents = text_loader.lazy_load()

# documents = pdf_documents + text_documents

# print(f"Total documents: {len(pdf_documents)}")

for document in pdf_documents:
    print("=" * 50)
    print("SOURCE:", document.metadata)
    # print(document.page_content[:500])