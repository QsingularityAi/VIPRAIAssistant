from langchain.llms import HuggingFaceEndpoint
from langchain.chat_models import ChatOpenAI
from groq import Groq
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
# os.environ["SAUERKRAUTLM_TGI_SERVER"] = "http://127.0.0.1:8080"
# def getLLM_SauerkrautLM():
    
#     inference_server_url_local = os.environ.get("SAUERKRAUTLM_TGI_SERVER") # For example: "http://127.0.0.1:8080"
#     myllm = HuggingFaceEndpoint(
#         endpoint_url=inference_server_url_local, # Corrected parameter name
#         max_new_tokens=512,
#         top_k=10,
#         top_p=0.95,
#         typical_p=0.95,
#         temperature=0.01,
#         repetition_penalty=1.03,
#     )
#     return myllm

def get_multimodal_llm():
    # Replace with your actual multimodal LLM configuration (e.g., OpenAI API)
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
    return model

def get_graq_model():
    llm = ChatGroq(temperature=1,
                      model_name="llama3-8b-8192",
                      api_key=os.environ.get("GROQ_API_KEY"))
    return llm
