
import time
from generate import *
from check import *
from process_data import *
from vectorDB import *
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def reliable_rag(urls:List[str], question:str, embedding_model = "nomic-embed-text:latest")-> str:
    """Main function to run the reliable RAG system.
    Args:
        question (str): User question
        docs (List): List of retrieved documents
    Returns:
        str: Answer to the question
    """
    # Load data
    docs_list = load_data(urls)

    # Build index
    docs_ojects = split_data(docs_list)
    vectordb = vectorstore(docs_ojects, collection_name="rag", embedding_model = embedding_model)
    retriver = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # number of documents to retrieve
            )
    # RAG
    docs = retriver.invoke(question)

    # Find the documents relevant
    docs_to_use = relevant_document(question, docs)

    # Generate answer
    generation = generate_answer(docs_to_use, question)

    # Check for hallucination
    response_check = check_halucination(docs_to_use, generation)
    print(f"Hallucination check: {response_check}\n")
    if response_check.binary_score == 'no':
        return "I don't know"
    else:
        print("Answer is grounded in the facts")

        # Highlight the specific part of a document used for answering the question
        lookup_response = highlight_doc(docs_to_use, question, generation)
        return lookup_response

if __name__ == "__main__":
    urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
    ]
    s = time.time()
    question = "what are the differnt kind of agentic design patterns?"
    answer = reliable_rag(urls,question)
    print(answer)
    print(f"Time taken: {time.time()-s} seconds")
