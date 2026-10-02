from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pathlib import Path

DATA_DIR = Path("books")
CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "my-rag-documents"

###### Load the documents from the directory
documents = []
for file_path in DATA_DIR.glob("*.txt"):
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    documents.extend(docs)
print(f"Loaded documents: {len(documents)}")

###### Update the metadata
for document in documents:
    source = Path(document.metadata["source"])
    document.metadata.update(
        {
            "filename":source.name,
            "document_type": "technical_document",
            "department": "AI",
            "access_level": "internal"
        }
    )

###### Text Splitting - making chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)
print(f"Loaded chunk: {len(chunks)}")

###### Indexing the chunks
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"]=index

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

####### Storing in Vector db - Chroma
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

vector_store.add_documents(chunks)
print("Documents successfully stored in ChromaDB")

##### querying and retrieving
query = "what is langchain"

results = vector_store.similarity_search(
    query=query,
    k=3
)

print("\n Retriever Documents")

for result in results:
    print("---------------------------------")
    print("Content:")
    print(result.page_content)
    print("----------------------------------")
    print("MetaData:")
    print(result.metadata)