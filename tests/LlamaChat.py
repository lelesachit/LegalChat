from llama_cpp import Llama
import os
import torch 
from langchain.vectorstores import Chroma
import chromadb
from transformers import pipeline
from langchain.vectorstores import chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

import threading


class LlamaChat:
    POLICY_COLL_NAME = "NLPProj"
    CHROMA_PATH = "./chromadb_data"
    client = chromadb.PersistentClient(
    path= CHROMA_PATH
)   
    embeddings = HuggingFaceInstructEmbeddings(
        model_name = "BAAI/bge-large-en",
        query_instruction = "Represent the request for legal acts related questions for retrieving of supporting information",
        embed_instruction = "Represent the request for legal document information for retrieval:" 
)

    trylock = threading.Lock()
    def __init__(self, model_path_or_repo_id:str, * , local_files_only = True, **kwargs):
        self.model_path_or_repo_id = model_path_or_repo_id
        self.llm = Llama(self.model_path_or_repo_id, n_gpu_layers= 5, n_ctx=4096)

        self.chrm = Chroma(LlamaChat.POLICY_COLL_NAME, persist_directory= LlamaChat.CHROMA_PATH, embedding_function= LlamaChat.embeddings)


    def generate(self, query:str, parameters:dict = None)->str:
        with LlamaChat.trylock:
            output = self.llm(query, max_tokens = 4096, temperature = 0.1)
            output = output['choices'][0]['text']
            return output