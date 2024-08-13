import os
import torch
from petals.constants import PUBLIC_INITIAL_PEERS
from chromadb.config import Settings

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

default_chat_config = ModelChatConfig(
    max_session_length=8192,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.6, top_p=0.9),
)

MODEL_FAMILIES = {
    
    
    "Llama 2": [
        ModelConfig(
            ModelBackendConfig(repository="meta-llama/Llama-2-70b-chat-hf"),
            ModelFrontendConfig(
                name="Llama 2 (70B-Chat)",
                model_card="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
                license="https://bit.ly/llama2-license",
            ),
            default_chat_config,
        ),
    ],
    
    "Mistral": [
         ModelConfig(
            ModelBackendConfig(repository="mistralai/Mixtral-8x7B-Instruct-v0.1"),
            ModelFrontendConfig(
                name="Mixtral 8x7B",
                model_card="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
                license="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/main",
            ),
            default_chat_config,
        ),
    ],
    
      
}
MIXTRAL_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# Define RAG Retrieval Settings
CHUNK_SIZE = 1000
TOP_K = 4
CHUNK_OVERLAP = 200

# Define RAG generation Settings
LLAMA2_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant, go through the context and answer the question strictly based on the context. 
<</SYS>>
Context: {context}
Question: {question} [/INST]"""

MIXTRAL_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant, go through the context and answer the question strictly based on the context. 
<</SYS>>
Context: {context}
Question: {question} [/INST]"""

LLAMA_PROMPT_TEMPLATE = """
### System:
You are a helpful, respectful and honest assistant, go through the context and answer the question strictly based on the context.

### User:
Context: {context}
Question: {question} 

### Assistant:
"""

# Define the Chroma settings
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS/text_data/text_data"
KAGGLE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
KAGGLE_DATASET = "rtatman/questionanswer-dataset"
STOP_TOKEN = "###"

HF_ACCESS_TOKEN = "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"
INITIAL_PEERS = []
BOOTSTRAP_PEERS = os.environ['INITIAL_PEERS']
if BOOTSTRAP_PEERS != "":
    bootstrap_list = BOOTSTRAP_PEERS.split(",")
    for peer in bootstrap_list:
        if peer != "":
            INITIAL_PEERS.append(BOOTSTRAP_PEERS)
#INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
# INITIAL_PEERS = ['/ip4/10.1.2.3/tcp/31234/p2p/QmcXhze98AcgGQDDYna23s4Jho96n8wkwLJv78vxtFNq44']

DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
GPU_INDEX = ""
try: 
    GPU_INDEX =  os.environ['DEVICE']
except Exception as e:
    pass 

if GPU_INDEX != "":
    DEVICE = GPU_INDEX

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE.startswith("cuda"):
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
