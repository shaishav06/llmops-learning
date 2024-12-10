# mypy: ignore-errors
import json
import os
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List

import mlflow
from guardrails import Guard
from guardrails.hub import RestrictToTopic
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

# Get the current working directory
script_dir = Path(os.getcwd())

# Navigate up to the parent folder (you can use .parent to go up one level)
parent_dir = script_dir.parent
grandparent_dir = parent_dir.parent  # Go up one more level

# Combine the path to reach the config directory
config_path = "rag_chain_config.yaml"

## Enable MLflow Tracing
mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config=config_path)

guardrail_config = model_config.get("guardrail_config")
llm_config = model_config.get("llm_config")
retriever_config = model_config.get("retriever_config")


# %% Setup Guardrails
# Import Guard and Validator


def get_topics_present(text: str, topics: List[str]) -> List[str]:
    """
    Determines which topics from a given list are present in the provided text using the ChatBedrock model.

    Args:
        text (str): The text in which to search for topics.
        topics (List[str]): A list of topics to check for presence in the text.

    Returns:
        List[str]: A list of topics that are present in the text. If no topics are found, returns an empty list.

    Example:
        >>> text = "Artificial intelligence and machine learning are transforming the tech industry."
        >>> topics = ["artificial intelligence", "machine learning", "blockchain"]
        >>> get_topics_present(text, topics)
        ['artificial intelligence', 'machine learning']
    """
    # Initialize the ChatBedrock model

    model = ChatBedrock(
        model_id=guardrail_config["model"],
        model_kwargs=dict(temperature=0.01),
    )

    # Prepare the messages for the model
    messages = [
        {
            "role": "system",
            "content": "Do not Output python functions. Given a conversation and a list of topics, return a valid JSON list of which topics are present in the final user question. If none, just return an empty list.",
        },
        {
            "role": "user",
            "content": f"""
                Given a text containing a user question, a chat_history and a list of topics, return a valid JSON list of which topics are present in the user question.
                Consider the entire history of the chat to determine the topics present in the user question. If the user question is unrelated with the chat history, consider only the user question.
                If none, just return an empty list.

                Output Format:
                -------------
                {{
                    "topics_present": []
                }}

                Text:
                ----
                "{text}"

                Topics: 
                ------
                {topics}

                Result:
                ------ 
            """,
        },
    ]

    # Invoke the model with the prepared messages
    response = json.loads(model.invoke(messages).content)
    return response["topics_present"]


# Setup Guard
guard = Guard().use(
    RestrictToTopic(
        valid_topics=guardrail_config["topics"]["valid"],
        invalid_topics=guardrail_config["topics"]["invalid"],
        llm_callable=get_topics_present,
        disable_classifier=True,  # Choose to use classifier or LLM
        disable_llm=False,
        on_fail="noop",
    )
)


def format_chat_history(messages):
    chat_history = ""
    for message in messages[:-1]:
        role = message["role"]
        content = message["content"]
        if role == "user":
            chat_history += f"User: {content}\n"
        elif role == "assistant":
            chat_history += f"Assistant: {content}\n"

    last_question = messages[-1]["content"]

    prompt = f"""You have been assigned with a task, based on the chat history below and the last user query.

    Chat history: {chat_history.strip()}

    Question: {last_question}"""

    return prompt


# Set Validation Function
def guardrail_the_input(messages):
    """
    Validates the input messages using guardrails.

    Args:
        messages (list): A list of message dictionaries representing the chat history.

    Returns:
        bool: True if the validation passes, False otherwise.
    """
    prompt = format_chat_history(messages)
    return guard.validate(prompt).validation_passed


# %% Setup Chatbot Chainß


# The question is the last entry of the history
def extract_question(input: List[Dict[str, Any]]) -> str:
    """
    Extract the question from the input.

    Args:
        input (list[dict]): The input containing chat messages.

    Returns:
        str: The extracted question.
    """
    return input[-1]["content"]


# The history is everything before the last question
def extract_history(input: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract the chat history from the input.

    Args:
        input (list[dict]): The input containing chat messages.

    Returns:
        list[dict]: The extracted chat history.
    """
    return input[:-1]


chat_model = ChatBedrock(
    model_id=llm_config["llm_model"],
    model_kwargs=dict(temperature=0.01),
)


def get_retriever(
    path: str = "http://localhost:6333",
):
    """
    Get the retriever.

    Args:
        path (str, optional): The path to the vector store. Defaults to None.

    Returns:
        Any: The retriever.
    """
    embeddings = BedrockEmbeddings()
    api_key = os.getenv("QDRANT_API_KEY")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=retriever_config.get("collection_name", "default"),
        url=path,
        api_key=api_key if api_key else None,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": retriever_config.get("parameters")["k"]}
    )

    return retriever


# Setup Prompt to re-write query from chat history context
generate_query_to_retrieve_context_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=llm_config["query_rewriter_prompt_template"],
)


# Setup query rewriter chain
generate_query_to_retrieve_context_chain = {
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
} | RunnableBranch(  # Augment query only when there is a chat history
    (
        lambda x: x["chat_history"],
        generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser(),
    ),
    (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
    RunnableLambda(lambda x: x["question"]),
)  # type: ignore


question_with_history_and_context_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=llm_config.get("llm_prompt_template"),  # Add Question with History and Context Prompt
)


def format_context(docs: List[Document]) -> str:
    """
    Format the context from a list of documents.

    Args:
        docs (list[Document]): A list of documents.

    Returns:
        str: A formatted string containing the content of the documents.
    """
    return "\n\n".join([d.page_content for d in docs])


def extract_source_urls(docs: List[Document]) -> List[str]:
    """
    Extract source URLs from a list of documents.

    Args:
        docs (list[Document]): A list of documents.

    Returns:
        list[str]: A list of source URLs extracted from the documents' metadata.
    """
    return [d.metadata[retriever_config.get("schema")["document_uri"]] for d in docs]


relevant_question_chain = (
    RunnablePassthrough()  # type: ignore
    | {
        "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser(),
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question"),
        "vector_store_path": itemgetter("vector_store_path"),
    }
    | {
        "relevant_docs": itemgetter("relevant_docs"),
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question"),
        "vector_store_path": itemgetter("vector_store_path"),
    }
    | RunnableLambda(
        lambda x: {
            "relevant_docs": get_retriever(x["vector_store_path"]).get_relevant_documents(
                x["relevant_docs"]
            ),
            "chat_history": x["chat_history"],
            "question": x["question"],
            "vector_store_path": x["vector_store_path"],
        }
    )
    | {
        "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
        "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question"),
    }
    | {"prompt": question_with_history_and_context_prompt, "sources": itemgetter("sources")}
    | {
        "result": itemgetter("prompt") | chat_model | StrOutputParser(),
        "sources": itemgetter("sources"),
    }
)


irrelevant_question_chain = RunnableLambda(
    lambda x: {"result": llm_config.get("llm_refusal_fallback_answer"), "sources": []}
)
branch_node = RunnableBranch(
    (lambda x: x["guardrails_validated"] == True, relevant_question_chain),  # noqa: E712
    (lambda x: x["guardrails_validated"] == False, irrelevant_question_chain),  # noqa: E712
    irrelevant_question_chain,  # Default
)  # type: ignore


full_chain = {
    "guardrails_validated": itemgetter("messages") | RunnableLambda(guardrail_the_input),
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    "vector_store_path": itemgetter("vector_store_path"),
} | branch_node  # type: ignore


## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=full_chain)  # type: ignore
