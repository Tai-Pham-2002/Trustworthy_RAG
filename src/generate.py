from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from typing import List

def format_docs(docs: List)-> str:
    """Format the documents to be used in the prompt.
    Args:
        docs (List): List of retrieved documents
    Returns:
        str: Formatted documents
    """
    return "\n".join(f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

def generate_answer(docs_to_use:List, question: str, model = "qwen3:4b"):
    """Generate answer from the retrieved documents.
    Args:
        docs_to_use (List): List of retrieved documents
        question (str): User question
    Returns:
        str: Generated answer
    """
    # Prompt
    system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge.
    Use three-to-five sentences maximum and keep the answer concise."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>./no_think"),
        ]
    )
    llm = ChatOllama(model=model, temperature=0, num_gpu=100)
    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"documents":format_docs(docs_to_use), "question": question})
    print(f"the generated answer is: {generation}\n")
    return generation