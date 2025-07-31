from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.output_parsers import PydanticOutputParser
from generate import format_docs
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from '.env' file
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') # For LLM -- qwen/qwen3-4b (small) & qwen/qwen3-32b (large)

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in 'generation' answer."""

    binary_score: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Data model
class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""

    id: List[str] = Field(
        ...,
        description="List of id of docs used to answers the question"
    )

    title: List[str] = Field(
        ...,
        description="List of titles used to answers the question"
    )

    source: List[str] = Field(
        ...,
        description="List of sources used to answers the question"
    )

    segment: List[str] = Field(
        ...,
        description="List of direct segements from used documents that answers the question"
    )


def relevant_document(question:str, docs:List)-> List:
    """Check if the retrieved documents are relevant to the question.
    Args:
        question (str): User question
        docs (List): List of retrieved documents
    Returns:
        List: List of relevant documents
    """
    # LLM with function call
    llm = ChatOllama(model="qwen3:4b", temperature=0, num_gpu=100)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}./no_think"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    docs_to_use = []

    for doc in docs:
        print(doc.page_content, '\n', '-'*50)
        res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        print(res,'\n')
        if res.binary_score == 'yes':
            docs_to_use.append(doc)
    
    return docs_to_use

def check_halucination(docs_to_use:List, generation:str):
    """Check if the retrieved documents are relevant to the question.
    Args:
        question (str): User question
        docs (List): List of retrieved documents
    Returns:
        response_check (List): List of hallucination check
    """
    # LLM with function call
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    response = hallucination_grader.invoke({"documents": format_docs(docs_to_use), "generation": generation})
    print(response)
    return response


def highlight_doc(docs_to_use:List, question:str, generation:str, model = "qwen/qwen3-32b")-> List:
    """Highlight the specific part of a document used for answering the question.
    Args:
        docs_to_use (List): List of retrieved documents
        question (str): User question
        generation (str): Generated answer
    Returns:
        List: List of highlighted documents
    """

    # LLM
    llm = ChatGroq(model=model, temperature=0)

    # parser
    parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

    # Prompt
    system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
    1. A question.
    2. A generated answer based on the question.
    3. A set of documents that were referenced in generating the answer.

    Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to
    generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text
    in the provided documents.

    Ensure that:
    - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
    - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
    - (Important) If you didn't used the specific document don't mention it.

    Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

    <format_instruction>
    {format_instructions}
    </format_instruction>
    """


    prompt = PromptTemplate(
        template= system,
        input_variables=["documents", "question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain
    doc_lookup = prompt | llm | parser

    # Run
    lookup_response = doc_lookup.invoke({"documents":format_docs(docs_to_use), "question": question, "generation": generation})
    return lookup_response