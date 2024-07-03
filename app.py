import hivemind
from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
import psutil
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

import config

#MADI

from transformers import AutoTokenizer, GenerationConfig, pipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callbacks = [StreamingStdOutCallbackHandler()]

logger = hivemind.get_logger(__file__)
#-----------------------------------
from transformers import TextStreamer,TextIteratorStreamer
from typing import Dict, Union, Any, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import tracing_enabled
from langchain.llms import OpenAI


# First, define custom callback handler implementations
class MyCustomHandlerOne(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        #print(f"on_llm_start {serialized['name']}")
        logger.info(f"on_llm_start: YES")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        logger.info(f"on_new_token {token}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        #print(f"on_chain_start {serialized['name']}")
        logger.info(f"on_chain_start: YES")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        #print(f"on_tool_start {serialized['name']}")
        logger.info(f"on_tool_start: YES")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        logger.info(f"on_agent_action {action}")


class MyCustomHandlerTwo(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"on_llm_start (I'm the second handler!!) {serialized['name']}")


# Instantiate the handlers
handler1 = MyCustomHandlerOne()
handler2 = MyCustomHandlerTwo()
#------------------------------------
os.environ['HF_ACCESS_TOKEN'] = "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"
#process.env.HF_ACCESS_TOKEN = 'hf_...';
access_token = "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"

models = {}
for model_info in config.MODELS:
    logger.info(f"Loading tokenizer for {model_info.repo}")
    tokenizer = AutoTokenizer.from_pretrained(model_info.repo, add_bos_token=False, use_fast=True,token=access_token,)

    logger.info(f"Loading model {model_info.repo} with adapter {model_info.adapter} and dtype {config.TORCH_DTYPE}")
    # We set use_fast=False since LlamaTokenizerFast takes a long time to init
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_info.repo,
        active_adapter=model_info.adapter,
        torch_dtype=config.TORCH_DTYPE,
        initial_peers=config.INITIAL_PEERS,
        max_retries=3,
        token=access_token,
    )
    model = model.to(config.DEVICE)
    generation_config = GenerationConfig.from_pretrained(model_info.repo)
    embeddings = HuggingFaceInstructEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": config.DEVICE})
    model_name = model_info.name
    if model_name is None:  # Use default name based on model/repo repo
        model_name = model_info.adapter if model_info.adapter is not None else model_info.repo
    models[model_name] = model,tokenizer,generation_config,embeddings #local_llm,embeddings

logger.info("Starting Flask app")
app = Flask(__name__)
CORS(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)


@app.route("/")
def main_page():
    return app.send_static_file("index.html")


import http_api
import websocket_api
