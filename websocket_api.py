import json
import os
from traceback import format_exc
from typing import List, Dict,Tuple

import flask_sock
import hivemind
from flask import request as http_request

import config
from app import models, sock
from utils import safe_decode

logger = hivemind.get_logger(__file__)
# RAG imports
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import torch
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

@sock.route("/api/v2/generate")
def ws_api_generate(ws):
    try:
        request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
        assert request["type"] == "open_inference_session"
        model_name = request["model"]
        if str(model_name).upper().find("LLAMA-2") ==-1 or str(model_name).upper().find("MIXTRAL") ==-1:
            model_name = config.MIXTRAL_MODEL_NAME
        max_length = request["max_length"]
        logger.info(f"ws.generate.open(), {model_name=}, {max_length=}, {http_request.origin=}")

        model, tokenizer, backend_config = models[model_name]
        if not backend_config.public_api and http_request.origin != f"{http_request.scheme}://{http_request.host}":
            raise ValueError(f"We do not provide public API for {model_name} due to license restrictions")

        with model.inference_session(max_length=max_length) as session:
            ws.send(json.dumps({"ok": True}))
            
            source_json = {}
            while True:
                request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
                assert request["type"] == "generate"
                inputs = request.get("inputs")

                logger.info(f"ws.generate.step(), inputs={repr(inputs)}")

                if inputs is not None:
                    logger.info(f"user_inputs = {inputs}")
                    #device_type = config.DEVICE if torch.cuda.is_available() else "cpu"
                    embeddings = HuggingFaceInstructEmbeddings(
                        model_name=config.EMBEDDING_MODEL_NAME,
                        model_kwargs={"device": config.DEVICE},
                    )
                    db = Chroma(
                        persist_directory=config.PERSIST_DIRECTORY,
                        embedding_function=embeddings,
                        client_settings=config.CHROMA_SETTINGS,
                    )
                    retriever = db.as_retriever(search_kwargs={'k': config.TOP_K})
                    docs = retriever.get_relevant_documents(query=inputs)
                    source_docs, context = fetch_contexts_and_sources(docs)
                    source_json = json.dumps(source_docs)
                    filled_prompt = ""
                    if str(model_name).upper().find("LLAMA-2") !=-1 or str(model_name).upper().find("MIXTRAL") !=-1:
                        prompt = PromptTemplate(input_variables=["context", "question"], template=config.LLAMA2_PROMPT_TEMPLATE)
                        filled_prompt = prompt.format(context=context,question=inputs)
                    else:
                        prompt = PromptTemplate(input_variables=["context", "question"], template=config.LLAMA_PROMPT_TEMPLATE)
                        filled_prompt = prompt.format(context=context,question=inputs)
                    
                    #logger.info(f"submitted_prompt = {filled_prompt}")      
                    inputs = tokenizer(filled_prompt, return_tensors="pt")["input_ids"].to(config.DEVICE)
                    n_input_tokens = inputs.shape[1]
                else:
                    n_input_tokens = 0

                stop_sequence = request.get("stop_sequence")
                extra_stop_sequences = request.get("extra_stop_sequences")
                if extra_stop_sequences is not None:
                    cont_token = tokenizer(stop_sequence, return_tensors="pt")["input_ids"].to(config.DEVICE)
                    if cont_token.shape != (1, 1):
                        raise ValueError("extra_stop_sequences require stop_sequence length to be exactly 1 token")

                all_outputs = ""
                delta_q = []
                stop = False
                while not stop:
                    outputs = model.generate(
                        inputs=inputs,
                        do_sample=request.get("do_sample", False),
                        temperature=request.get("temperature"),
                        top_k=request.get("top_k"),
                        top_p=request.get("top_p"),
                        repetition_penalty=request.get("repetition_penalty"),
                        max_length=request.get("max_length"),
                        max_new_tokens=request.get("max_new_tokens"),
                        session=session,
                    )
                    delta = outputs[0, n_input_tokens:].tolist()
                    outputs = safe_decode(tokenizer, delta_q + delta)
                    inputs = None  # Inputs are passed only for the 1st token of the bot's response
                    n_input_tokens = 0
                    combined = all_outputs + outputs
                    stop = stop_sequence is None or (
                        "falcon-180B" not in model_name and combined.endswith(stop_sequence)
                    )
                    if extra_stop_sequences is not None:
                        for seq in extra_stop_sequences:
                            if combined.endswith(seq):
                                stop = True
                                session.last_token_id = cont_token
                    if not stop and outputs[-10:].find("\ufffd") > -1:
                        # If there's a replacement character, keep getting more tokens
                        # until we can decode properly
                        delta_q = delta_q + delta
                        logger.info(f"ws.generate.append_retry(), all_outputs={repr(combined)}")
                    else:
                        all_outputs = combined
                        token_count = len(delta_q + delta)
                        delta_q = []
                        logger.info(f"ws.generate.step(), all_outputs={repr(all_outputs)}, stop={stop}")
                        #HIVE START
                        sessionlist = session._server_sessions
                        route_map = {}
                        for sid in sessionlist:
                            block_range = str(sid.span.start) + ":" + str(sid.span.end)
                            ip_addr = str(sid.span.server_info.public_name)
                            peer_id = str(sid.span.peer_id)
                            route_map[block_range] = ip_addr + " (..." + peer_id[-5:] +")"
                            #logger.info(f"HIVE: PeerID = {sid.span.peer_id}; BLOCKS = {sid.span.start},{sid.span.end}")
                        route_json = json.dumps(route_map)
                        #HIVE END
                        ws.send(json.dumps({"ok": True, "outputs": outputs, "stop": stop, "token_count": token_count, "route":route_json, "source_documents": source_json}))
                        #logger.info(f"source_docs = {source_json}")
    except flask_sock.ConnectionClosed:
        pass
    except Exception:
        logger.warning("ws.generate failed:", exc_info=True)
        ws.send(json.dumps({"ok": False, "traceback": format_exc()}))
    finally:
        logger.info(f"ws.generate.close()")
        

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
                