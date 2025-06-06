from langchain_core import prompts

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_community.vectorstores import AzureSearch

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError



from pathlib import Path
import json

from typing import List,Dict

from qdrant import VectorStore

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.json"

DEFAULT_DIMS = 3072

def get_openai_models(dimensions = DEFAULT_DIMS, temperature = 0 ):
    """
    @return Instance of AzureChatOpenAI.
            Instance of AzureOpenAIEmbeddings.
    """

    config_path  = Path(CONFIG_PATH)

    try:
        with config_path.open("r") as f:
            config = json.load(f)

    except FileNotFoundError:
        print("[Error] config.json not found.")

    azure_config = config.get("azure_openai", {})
    azure_search_config = config.get("azure_search", {})

    if not azure_config:
        print("[Error] Failed to load azure configuration.")
        return None, None, None, False    

    llm = AzureChatOpenAI(
        azure_endpoint = azure_config["azure_endpoint"],
        api_key = azure_config["api_key"],
        api_version = azure_config["api_version"],

        azure_deployment = azure_config["chat_deployment"]["gpt-4.1"],

        temperature = temperature
    )

    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint = azure_config["azure_endpoint"],
        api_key = azure_config["api_key"],
        api_version = azure_config["api_version"],

        azure_deployment = azure_config["embedding_deployment"]["large"],
        dimensions = dimensions
    )


    search_endpoint = azure_search_config.get("azure_search_endpoint", "")
    search_key = azure_search_config.get("azure_search_key", "")
    index_name = azure_search_config.get("index_name", "")

    if search_endpoint and search_key and index_name:
        print("[Msg] Using Azure AI Search as vector store.")
        vector_store = AzureSearch(
            azure_search_endpoint = search_endpoint,
            azure_search_key = search_key,
            index_name = index_name,
            embedding_model = embedding_model
        )
        using_local_store = False
    else:
        print("[Msg] Azure AI Search config missing, using local vector store.")
        vector_store = VectorStore(embedding_model = embedding_model)

        using_local_store = True

    return llm, embedding_model, vector_store, using_local_store 


def chat_bot(query:str) -> str:

    llm, embedding_model, vector_store, using_local_store = get_openai_models()

    
    if llm is None:
        return "[Error] LLM not available. Please check your configuration."
    
    if embedding_model is None:
        return "[Error] Embedding model unavailable. Please check your configuration."
    
    if vector_store is None:
        return "[Error] Vectore store unavailable. Please check your configuration."
    

    # Set up Chatbot
    chatbot_prompt = prompts.PromptTemplate.from_template("""

        You're a smart assistant helping extract insights from VTT innovation relationships.

        Context:
        {context}

        According to the context, answer this question:
        {question}
    """)

    chatbot_llm = chatbot_prompt | llm

    if using_local_store:

        try:
            _, result_texts = vector_store.search([query], limit = 5)
            context = "\n".join(result_texts)

        except Exception as e:
            print(f"[Error] during vector search: {str(e)}")
            return "[Error] Local vector search failed."


        try:
            llm_result = chatbot_llm.invoke({"context":context, "question":query})
            answer = llm_result.content

        except Exception as e:
            print(f"[Error] generating answer: {str(e)}")
            return f"[Error] LLM failed process your question. Error: {str(e)}"


    else:
        try:
            results = vector_store.vector_search(query, k = 5)
            context = "\n".join([res.metadata['source'] for res in results])

        except Exception as e:
            print(f"[Error] during vector search: {str(e)}")
            return "[Error] Azure AI search: vector search failed."

        try:
            llm_result = chatbot_llm.invoke({"context":context, "question":query})
            answer = llm_result.content

        except Exception as e:
            print(f"[Error] generating answer: {str(e)}")
            return f"[Error] LLM failed process your question. Error: {str(e)}"
    
    return answer



