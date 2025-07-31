### LLMs
import os
from dotenv import load_dotenv
### Build Index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
from typing import List


def load_data(urls:List[str])-> List:
    """Upload the documents to the vector store.
    Args:
        urls (List): List of urls
    Returns:
        List: List of documents
    """
    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def split_data(docs_list)-> List:
    """Split the documents into chunks.
    Args:
        docs_list (List): List of documents
    Returns:
        List: List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

