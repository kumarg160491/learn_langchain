from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

# 1. Define your mixed-topic documents
docs = [
    Document(
        page_content=(
            "The James Webb Space Telescope recently captured stunning infrared images of the Pillars of Creation. "
            "These high-resolution snapshots reveal new stars forming inside dense clouds of interstellar gas and dust, "
            "providing scientists with deeper insights into stellar life cycles."
        ),
        metadata={'source': "Doc1"},
    ),
    Document(
        page_content=(
            "When baking sourdough bread, the Maillard reaction is responsible for creating the deep, dark brown crust. "
            "This chemical reaction occurs between amino acids and reducing sugars at high temperatures, "
            "giving the bread its characteristic complex flavors and rich aromas."
        ),
        metadata={'source': "Doc2"},
    ),
    Document(
        page_content=(
            "Modern workplace studies show that cognitive fatigue decreases when employees take structured micro-breaks. "
            "Stepping away from a computer screen for just five minutes every hour helps reset focus, "
            "lowers stress hormones, and significantly boosts creative problem-solving abilities."
        ),
        metadata={'source': "Doc3"},
    ),
]

# 2. Setup your embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# 🛠️ FIX: Use .from_documents() instead of directly calling FAISS()
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# 3. Create the base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# 4. Setup your local LLM and compressor
llm = ChatOllama(model="gemma2:2b-instruct-q5_0")
compressor = LLMChainExtractor.from_llm(llm)

# 5. Create your compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 6. Set up your test query
query = "What did the James Webb Space Telescope capture in the Pillars of Creation?"

# 🛠️ FIX: Query the compression retriever instead of printing raw docs
compressed_docs = compression_retriever.invoke(query)

# 7. Print the results to see the compression in action
print(f"Query: {query}\n")
for i, doc in enumerate(compressed_docs):
    print(f'---- Compressed Result {i+1} (Source: {doc.metadata["source"]}) ------')
    print(doc.page_content)
