import os
from traceback import format_exc

import hivemind
from flask import jsonify, request

import config
from app import app, models
from utils import safe_decode
from werkzeug.utils import secure_filename

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

logger = hivemind.get_logger(__file__)
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"


# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

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

#Houssam
import hivedisk_api

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS/"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# Houssam Import



@app.post("/api/v1/gethivedisk")
def update_from_hiveDisk():
    file_list = hivedisk_api.list_all_files()
    logger.info(f"HiveDisk files List: {file_list}")
    hivedisk_api.get_files(filelist=file_list,path=SOURCE_DIRECTORY)
    logger.info(f"HiveDisk files Downloaded into: {SOURCE_DIRECTORY} !!")
    # Load documents and split in chunks
    logger.info(f"Loading HiveDisk documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logger.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logger.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    device_type = "cuda:1" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,

    )
    logger.info(f"Knowledge DB Updated with HiveDisk Data !!")
    return "OK"
    

@app.post("/api/v1/updatedb")
def http_api_update_db():
    uploaded_files = request.files.getlist('file')
    allok = True
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        if filename != '':
            file.save(os.path.join(SOURCE_DIRECTORY, filename))
    
    # Load documents and split in chunks
    logger.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logger.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logger.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    device_type = "cuda:1" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,

    )

    return "OK"


@app.post("/api/v1/generate")
def http_api_generate():
    try:
        model_name = get_typed_arg("model", str, config.DEFAULT_MODEL_NAME)
        inputs = request.values.get("inputs")
        do_sample = get_typed_arg("do_sample", int, False)
        temperature = get_typed_arg("temperature", float)
        top_k = get_typed_arg("top_k", int)
        top_p = get_typed_arg("top_p", float)
        repetition_penalty = get_typed_arg("repetition_penalty", float)
        max_length = get_typed_arg("max_length", int)
        max_new_tokens = get_typed_arg("max_new_tokens", int)
        session_id = request.values.get("session_id")
        logger.info(f"generate(), model={repr(model_name)}, inputs={repr(inputs)}")

        if session_id is not None:
            raise RuntimeError(
                "Reusing inference sessions was removed from HTTP API, please use WebSocket API instead"
            )

        model, tokenizer, generation_config,embeddings = models[model_name]

        #if inputs is not None:
            #inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            #n_input_tokens = inputs.shape[1]
        #else:
            #n_input_tokens = 0
        max_ctx_size = 2048 #2048
        kwargs = {
            "n_ctx": max_ctx_size,
            "max_tokens": max_ctx_size,
            "n_threads": psutil.cpu_count(logical=False),
            "max_tokens": max_ctx_size
        }
        
        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        model_kwargs=kwargs,
        use_fast=True,
        max_new_tokens=50,
        do_sample=False,
        #use_cache=False,
        device=torch.device('cuda:1') #config.DEVICE #"cuda:0"
        )
        #local_llm = HuggingFacePipeline(pipeline=pipe,callbacks=callbacks)
        local_llm = HuggingFacePipeline(pipeline=pipe)
        #embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": config.DEVICE})
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
        #a = StreamingStdOutCallbackHandler()
        #a.on_llm_new_token()
        #callbacks.append().on_llm_new_token()
        retriever = db.as_retriever(search_kwargs={'k': 4})
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
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}#, "memory": memory},
        )
        res = qa(inputs)
        answer, docs = res["result"], []
        '''
        outputs = model.generate(
            inputs=inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        outputs = safe_decode(tokenizer, outputs[0, n_input_tokens:])
        '''
        logger.info(f"generate(), outputs={repr(answer)}")
        topAnswer = answer.split("Question")[0].strip()
        combined = repr(topAnswer)
        logger.info(f"generate(), cleanoutputs={combined}")
        stop = True

        return jsonify(ok=True, outputs=topAnswer)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


def get_typed_arg(name, expected_type, default=None):
    value = request.values.get(name)
    return expected_type(value) if value is not None else default

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        try:
            loader = loader_class(file_path)
        except Exception as e:
            logger.warning(f"ignoring a malformed file, filename: {file_path}, err: {e}")
            raise(e)
            
    else:
        raise ValueError("Document type is undefined")
    try: 
        return loader.load()[0]
    except Exception as e:
        logger.warning(f"ignoring a malformed file, filename: {file_path}, err: {e}")
        raise(e)


def load_document_batch(filepaths):
    logger.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as e:
                logger.warning(f"ignoring a malformed file, filename: {future.exception().args[0]}, err: {e}")
                continue

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs
