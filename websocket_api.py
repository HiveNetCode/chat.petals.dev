import json
import threading
import time
from traceback import format_exc

import flask_sock
import hivemind
import torch

import config
from app import sock, models
from utils import safe_decode
import re

#MADI
from transformers import TextStreamer,TextIteratorStreamer
from typing import Dict, Union, Any, List
import psutil
from transformers import AutoTokenizer, GenerationConfig, pipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import os
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate

logger = hivemind.get_logger(__file__)

logger = hivemind.get_logger(__file__)

INFERENCE_PATH = {} #manager.dict()
GLOBAL_MAP = {}
GLOBAL_NAME = ""
lock = threading.Lock()
isDummyRunning = False

from langchain_core.callbacks import BaseCallbackHandler

class StoreDocumentsCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.source_documents = {}

    def on_retriever_end(self, documents, **kwargs):
        #self.source_documents.extend(documents)
        temp ={}
        for doc in documents:
            filename = os.path.basename(doc.metadata['source'])
            if  self.source_documents.get(filename) is not None:
                temp[filename] = temp[filename] +1
                self.source_documents[filename + " (" + str(temp[filename]) + ")"] = doc.page_content
            else:
               temp[filename] = 0
               self.source_documents[filename] = doc.page_content 
                
            #self.source_documents[os.path.basename(doc.metadata['source'])] = doc.page_content
        logger.info(f"HIVE: CALLBACK SRC_DOCS: {self.source_documents}")
            

def run_dummy_session(model,tokenizer,name):
    
    global GLOBAL_MAP
    global GLOBAL_NAME
    #with lock:
    #    if len(GLOBAL_MAP) > 0:
    #        logger.info(f"HIVE: DummyPath via: {GLOBAL_MAP}")
    #        return
    with model.inference_session(max_length=25) as session:
        GLOBAL_MAP = {}
        while True:
            found = False
            inputs = "hi"
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            #n_input_tokens = inputs.shape[1]
            _ = model.generate(inputs=inputs,do_sample=False,max_new_tokens=1,session=session)
            sessionlist = session._server_sessions
                
            for sid in sessionlist:
                found = True
                block_range = str(sid.span.start) + ":" + str(sid.span.end)
                ip_addr = str(sid.span.server_info.public_name)
                peer_id = str(sid.span.peer_id)
                with lock:
                    GLOBAL_MAP[block_range] = ip_addr + " (..." + peer_id[-5:] +")"
            if found:
                logger.info(f"HIVE: DummyPath via: {GLOBAL_MAP}")
                GLOBAL_NAME = name
                return

@sock.route("/api/v2/generate")
def ws_api_generate(ws):
    source_json = {}
    try:
        request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
        assert request["type"] == "open_inference_session" #"generate"   
        #"open_inference_session"


        model_name = config.DEFAULT_MODEL_NAME #
        model_name = request.get("model")
        if model_name is None:
            model_name = config.DEFAULT_MODEL_NAME
        logger.info(f"ws.generate.open(), model={repr(model_name)}, max_length={repr(request['max_length'])}")

        model, tokenizer,generation_config,embeddings = models[model_name]
        global GLOBAL_MAP
        global GLOBAL_NAME
        if len(GLOBAL_MAP) <= 1 or GLOBAL_NAME != model_name:
            GLOBAL_MAP = {}
            dummySession = threading.Thread(target=run_dummy_session, args=(model,tokenizer,model_name))
            dummySession.start()
        #isDummyRunning = True

        ws.send(json.dumps({"ok": True}))
        
        #global isDummyRunning
        #if not isDummyRunning:
            #mp.set_start_method('spawn')
        
        
        request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
        assert request["type"] == "generate"
        inputs = request.get("inputs")
        
        logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
        n_input_tokens = 0
        nm_tokens = 0
        if inputs is not None:
            
            
            temp0 = repr(inputs).split("###Human:")
            temp1 = ""
            UserInput = ""
            if len(temp0)> 0:
                temp1 = temp0[len(temp0)-1].split("###")
                if len(temp1) > 0:
                    UserInput = temp1[0].strip()
                    UserInput = UserInput.replace('Human:', '')
            logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
            logger.info(f"ws.generate.step(), UserInput={repr(UserInput)}")
            inputs = UserInput
            nm_tokens = len(inputs.split())
            
        else:
            n_input_tokens = 0
        
        max_ctx_size = 4096 #2048
        kwargs = {
            "n_ctx": max_ctx_size,
            "max_tokens": max_ctx_size,
            "n_threads": psutil.cpu_count(logical=False),
            "max_tokens": max_ctx_size
        }
        streamer = TextIteratorStreamer(tokenizer, timeout=30., skip_prompt=True, skip_special_tokens=True)
        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        config={'max_length': 4096},
        generation_config=generation_config,
        model_kwargs=kwargs,
        use_fast=True,
        max_new_tokens=1024,
        do_sample=False,
        streamer=streamer,
        #use_cache=False,
        device=config.DEVICE #config.DEVICE #"cuda:0"
        )
        #pipe.streamer = streamer
        #local_llm = HuggingFacePipeline(pipeline=pipe,callbacks=callbacks)
        local_llm = HuggingFacePipeline(pipeline=pipe)
        #embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": config.DEVICE})
        db = Chroma(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=config.CHROMA_SETTINGS,
        )
        #a = StreamingStdOutCallbackHandler()
        #a.on_llm_new_token()
        #callbacks.append().on_llm_new_token()
        retriever = db.as_retriever(search_kwargs={'k': 3})
        # Create a system prompt 
        template = """You are a helpful, respectful and honest assistant. Always answer as 
        helpfully and naturally as possible, while being safe.
        
        Your goal is to provide answers based strictly on the following pieces of context. Go through the provided context and answer the given question strictly based on the provided contexts. If you cannot determine the answer from the provided context, please say that you don't know. 
        Ensure your answer is clear and free of any metadata tags, special characters, or non-human readable characters. Please ansewr directly without stating you are answering.
        

        {context}

        Question: {question}
        Answer:"""

        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        memory = ConversationBufferMemory(input_key="question", memory_key="history")
        
        callback = StoreDocumentsCallback()
        #callback_manager = CallbackManager([callback])
        qa = RetrievalQA.from_chain_type(
            llm=local_llm, #model,
            chain_type="stuff",
            retriever=retriever, #reduce_k_below_max_tokens=True,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}#,#, "memory": memory},
            #callbacks=[callback] 
            #callback_manager=callback_manager
        )
        def run_enhanced_rqa(message):
            qa.run(message,callbacks=[callback])

        t = threading.Thread(target=run_enhanced_rqa, args=(UserInput,))
        t.start()

        max_token = 1024
        index = 0 
        stop = False
        sequence = ""
        source_json = {}
        while True:
            for outputs in streamer:
                sequence +=outputs
                if ((sequence.find("\n\n\n\n")!=-1) or ( (len(sequence)>5) and (sequence.isspace()))):
                    stop = True
                    index = 0
                #global GLOBAL_MAP
                token_count = 0
                route_json = {}
                source_json = {}
                source_json = json.dumps(callback.source_documents)
                #time.sleep(0.05)
                with lock:
                    route_json = json.dumps(GLOBAL_MAP)
                #HIVE END
                token_count = len(outputs.split())
                stop_sequence = request.get("stop_sequence")
                if ((outputs.endswith(stop_sequence)) or (outputs.endswith("\n\n\n\n")) or (index >= max_token)):
                    stop = True
                if ((outputs.endswith("Question")) or (outputs.find("Question")!=-1)):
                    stop = True
                    index = 0
                    outputs = ""
    
                if index >= max_token:
                    stop = True
                  
                if stop and outputs.isspace():
                    outputs = "Sorry, I would need to learn more.\n"
                    token_count = 12
                    ws.send(json.dumps({"ok": True, "outputs": "Sorry, I would need to learn more.\n", "stop": True, "token_count": 12, "route":route_json, "source_documents": source_json}))
                    #logger.info(f"source_docs = {source_json}")
                ws.send(json.dumps({"ok": True, "outputs": outputs, "stop": stop, "token_count": token_count, "route":route_json, "source_documents": source_json}))
                logger.info(f"source_docs = {source_json}")
                incr = len(outputs.split())
                index+=incr
                logger.info(f"HIVE Incr Ouptput = {outputs}")
                #logger.info(f"source_docs = {source_json}")

                if stop:
                    index = 0
                    break
            if stop:
                index = 0
                break
    except flask_sock.ConnectionClosed:
        pass
    except Exception:
        logger.warning("ws.generate failed:", exc_info=True)
        ws.send(json.dumps({"ok": True, "outputs": "\n", "stop": True, "token_count": 1, "route":json.dumps(GLOBAL_MAP), "source_documents": source_json}))
        #logger.info(f"source_docs = {source_json}")
        #ws.send(json.dumps({"ok": False, "traceback": format_exc()}))
    finally:
        logger.info(f"ws.generate.close()")
        

# Function to truncate context to a maximum token length
def truncate_to_fit(context, question, max_tokens):
    question_tokens = question.split()
    context_tokens = context.split()
    
    if len(question_tokens) + len(context_tokens) > max_tokens:
        context_tokens = context_tokens[:max_tokens - len(question_tokens)]
    
    return ' '.join(context_tokens)
