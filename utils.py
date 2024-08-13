from typing import Dict, List, Tuple, Union

import hivemind
import torch
from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from langchain.docstore.document import Document
import config
from data_structures import ModelConfig
import os

logger = hivemind.get_logger(__file__)


def load_models() -> Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer, ModelConfig]]:
    models = {}
    for family in config.MODEL_FAMILIES.values():
        for model_config in family:
            backend_config = model_config.backend

            logger.info(f"Loading tokenizer for {backend_config.repository}")
            tokenizer = AutoTokenizer.from_pretrained(backend_config.repository, add_bos_token=False, use_fast=False,token=config.HF_ACCESS_TOKEN,)

            logger.info(
                f"Loading model {backend_config.repository} with adapter {backend_config.adapter} in {config.TORCH_DTYPE}"
            )
            # We set use_fast=False since LlamaTokenizerFast takes a long time to init
            model = AutoDistributedModelForCausalLM.from_pretrained(
                backend_config.repository,
                active_adapter=backend_config.adapter,
                torch_dtype=config.TORCH_DTYPE,
                initial_peers=config.INITIAL_PEERS,
                max_retries=3,
                token=config.HF_ACCESS_TOKEN,
            )
            model = model.to(config.DEVICE)

            for key in [backend_config.key] + list(backend_config.aliases):
                models[key] = model, tokenizer, backend_config
    return models


def safe_decode(tokenizer: PreTrainedTokenizer, outputs: Union[torch.Tensor, List[int]]) -> str:
    # Workaround to make SentencePiece .decode() keep leading spaces in a token
    fake_token = tokenizer("^")["input_ids"][0]
    outputs = outputs.tolist() if isinstance(outputs, torch.Tensor) else outputs
    result = tokenizer.decode([fake_token] + outputs)

    # We use .lstrip() since SentencePiece may add leading spaces, e.g. if the outputs are "</s>"
    return result.lstrip()[1:]

def fetch_contexts_and_sources(documents: List[Document]) -> Tuple[Dict[str, str], str]:
        #self.source_documents.extend(documents)
        temp ={}
        source_documents = {}
        context_list = []
        for doc in documents:
            filename = os.path.basename(doc.metadata['source'])
            if  source_documents.get(filename) is not None:
                temp[filename] = temp[filename] +1
                source_documents[filename + " (" + str(temp[filename]) + ")"] = doc.page_content
                context_list.append(doc.page_content)
            else:
               temp[filename] = 0
               source_documents[filename] = doc.page_content
               context_list.append(doc.page_content)
        context = "\n".join(context_list)
        return source_documents,context
