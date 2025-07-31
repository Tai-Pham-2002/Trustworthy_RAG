from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def vectorstore(docs, collection_name="rag", embedding_model = "nomic-embed-text:latest")-> Chroma:
    """Create a vectorstore from the documents.
    Args:
        docs (List): List of documents
        collection_name (str): Name of the collection
        embedding_model (str): Name of the embedding model
    Returns:
        Chroma: Chroma vectorstore
    """
    embedding = OllamaEmbeddings(model=embedding_model, num_gpu=100)
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name=collection_name,
        embedding=embedding,
        )
    return vectorstore