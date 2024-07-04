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
        
        max_ctx_size = 2048 #2048
        kwargs = {
            "n_ctx": max_ctx_size,
            "max_tokens": max_ctx_size,
            "n_threads": psutil.cpu_count(logical=False),
            "max_tokens": max_ctx_size
        }
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        config={'max_length': 2048},
        generation_config=generation_config,
        model_kwargs=kwargs,
        use_fast=True,
        max_new_tokens=1024,
        do_sample=False,
        streamer=streamer,
        #use_cache=False,
        device=torch.device('cuda') #config.DEVICE #"cuda:0"
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
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
        just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""

        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        memory = ConversationBufferMemory(input_key="question", memory_key="history")

        qa = RetrievalQA.from_chain_type(
            llm=local_llm, #model,
            chain_type="stuff",
            retriever=retriever, #reduce_k_below_max_tokens=True,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}#, "memory": memory},
        )
        def run_enhanced_rqa(message):
            qa.run(message)

        t = threading.Thread(target=run_enhanced_rqa, args=(UserInput,))
        t.start()

        max_token = 1024
        index = 0 
        stop = False
        sequence = ""
        while True:
            for outputs in streamer:
                sequence +=outputs
                if ((sequence.find("\n\n\n\n")!=-1) or ( (len(sequence)>5) and (sequence.isspace()))):
                    stop = True
                    index = 0
                #global GLOBAL_MAP
                token_count = 0
                route_json = {}
                #time.sleep(0.05)
                with lock:
                    route_json = json.dumps(GLOBAL_MAP)
                #HIVE END
                token_count = len(outputs.split())
                stop_sequence = request.get("stop_sequence")
                if ((outputs.endswith(stop_sequence)) or (outputs.endswith("\n\n\n\n")) or (index >= max_token)):
                    stop = True
                #    index = 0
                    #outputs = ""
                    #token_count = 0 
                #if ((outputs.endswith("-----")) or (outputs.find("-----")!=-1)):
                #    stop = True
                    #outputs = ""
                    #token_count = 0
                if ((outputs.endswith("Question")) or (outputs.find("Question")!=-1)):
                    stop = True
                    index = 0
                    outputs = ""
                #if outputs == "":
                #    stop = True
                if index >= max_token:
                    stop = True
                    #outputs = ""
                    #token_count = 0
                #if index > nm_tokens-1: #(n_input_tokens-5):
                if stop and outputs.isspace():
                    outputs = "Sorry, I would need to learn more.\n"
                    token_count = 12
                    ws.send(json.dumps({"ok": True, "outputs": "Sorry, I would need to learn more.\n", "stop": True, "token_count": 12, "route":route_json}))
                ws.send(json.dumps({"ok": True, "outputs": outputs, "stop": stop, "token_count": token_count, "route":route_json}))
                incr = len(outputs.split())
                index+=incr
                logger.info(f"HIVE Incr Ouptput = {outputs}")

                if stop:
                    index = 0
                    break
            if stop:
                index = 0
                break
            #outputs = [text]
            '''
                res = qa(UserInput)#qa(inputs)
                answer, docs = res["result"], []
                topAnswer = answer.split("\n")[0]
                combined = repr(answer)
                stop = True
                #stop = stop_sequence is None or combined.endswith(stop_sequence)
                #if extra_stop_sequences is not None:
                #    for seq in extra_stop_sequences:
                #        if combined.endswith(seq):
                #            stop = True
                if stop:
                    logger.info(f"ws.generate.step(), all_outputs={answer}, stop={stop}")
                    token_count = len(combined.split())
                    # Use regular expressions to remove lines starting with 'Question' or 'Answer'
                    clean_answer = re.sub(r'^\s*(Question:|Answer:|Question|Answer).*$', '', answer, flags=re.MULTILINE)
                    #print(text_without_questions_and_answers)
                    global GLOBAL_MAP
                    #token_count = 0
                    route_json = {}
                    with lock:
                        route_json = json.dumps(GLOBAL_MAP)
                    
                    logger.info(f"ROUTE: DummyPath via: {route_json}")
                    #logger.info(f"CALLBACK CONTENT: {callbacks.pop()}")
                    ws.send(json.dumps({"ok": True, "outputs": clean_answer.strip(), "stop": stop, "token_count": token_count,"route":route_json}))
        
        with model.inference_session(max_length=request["max_length"]) as session:
            ws.send(json.dumps({"ok": True}))
            inputs = request.get("inputs")
            if inputs is not None:
                logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
                res = qa(inputs)
                answer, docs = res["result"], []
                stop = True
                logger.info(f"ws.generate.step(), all_outputs={repr(answer)}, stop={stop}")
                ws.send(json.dumps({"ok": True, "outputs": answer, "stop": stop, "token_count": 0}))
        '''
        
        '''
            while True:
                request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
                assert request["type"] == "generate"
                inputs = request.get("inputs")
                logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
                
                if inputs is not None:
                    inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
                    n_input_tokens = inputs.shape[1]
                else:
                    n_input_tokens = 0
                

                stop_sequence = request.get("stop_sequence")
                extra_stop_sequences = request.get("extra_stop_sequences")
                #if extra_stop_sequences is not None:
                    #cont_token = tokenizer(stop_sequence, return_tensors="pt")["input_ids"].to(config.DEVICE)
                    #assert cont_token.shape == (1, 1), \
                    #    "extra_stop_sequences require stop_sequence length to be exactly 1 token"

                all_outputs = ''
                delta_q = []
                stop = False
                if not stop:
                    res = qa(inputs)
                    
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
                    
                    answer, docs = res["result"], []
                    #delta = answer[0, n_input_tokens:].tolist()
                    #outputs = safe_decode(tokenizer, delta_q + delta)
                    inputs = None  # Inputs are passed only for the 1st token of the bot's response
                    n_input_tokens = 0
                    combined = all_outputs + answer #outputs
                    #stop = stop_sequence is None or combined.endswith(stop_sequence)
                    stop = True
                    
                    if extra_stop_sequences is not None:
                        for seq in extra_stop_sequences:
                            if combined.endswith(seq):
                                stop = True
                                session.last_token_id = cont_token
                    
                    if not stop:
                        # If there's a replacement character, keep getting more tokens
                        # until we can decode properly
                        #delta_q = delta_q + delta
                        logger.info(f"ws.generate.append_retry(), all_outputs={repr(combined)}")
                    else:
                        all_outputs = combined
                        #token_count = len(delta_q + delta)
                        #delta_q = []
                        logger.info(f"ws.generate.step(), all_outputs={repr(all_outputs)}, stop={stop}")
                        ws.send(json.dumps({"ok": True, "outputs": answer, "stop": stop, "token_count": 0}))
                        '''
    except flask_sock.ConnectionClosed:
        pass
    except Exception:
        logger.warning("ws.generate failed:", exc_info=True)
        ws.send(json.dumps({"ok": True, "outputs": "\n", "stop": True, "token_count": 1, "route":json.dumps(GLOBAL_MAP)}))
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

