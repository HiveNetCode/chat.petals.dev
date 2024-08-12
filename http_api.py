from traceback import format_exc

import hivemind
from flask import jsonify, request

import config
from app import app, models
from utils import safe_decode
import json
import os
# imports for RAG
from werkzeug.utils import secure_filename
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

from kaggle.api.kaggle_api_extended import KaggleApi

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".clean": TextLoader,
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

from ragas import evaluate
from datasets import Dataset
from ragas.metrics.critique import harmfulness
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from fastapi import FastAPI, Request
#Houssam
import hivedisk_api
import numpy as np # linear algebra
import pandas as pd 

logger = hivemind.get_logger(__file__)


@app.post("/api/v1/gethivedisk")
def update_from_kaggle():
    # Initialize the Kaggle API
    kaggleApi = KaggleApi()
    kaggleApi.authenticate()
    logger.info(f"kaggle authentication OK")
    logger.info(f"Downloading [{config.KAGGLE_DATASET}] from kaggle into: {config.KAGGLE_DIRECTORY} ...")
    kaggleApi.dataset_download_files(config.KAGGLE_DATASET, path=config.KAGGLE_DIRECTORY, unzip=True)
    logger.info(f"Kaggle dataset Downloaded into: {config.KAGGLE_DIRECTORY} !!")
    # Load documents and split in chunks
    logger.info(f"Loading Kaggle documents from {config.SOURCE_DIRECTORY}")
    documents = load_documents(config.SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=412, chunk_overlap=50
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logger.info(f"Loaded {len(documents)} documents from {config.SOURCE_DIRECTORY}")
    logger.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    device_type = config.DEVICE if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIRECTORY,
        client_settings=config.CHROMA_SETTINGS,

    )
    logger.info(f"Knowledge DB Updated with Kaggle Dataset !!")
    return "OK"

@app.post("/api/v1/generate")
def http_api_generate():
    try:
        model_name = get_typed_arg("model", str)
        inputs = request.values.get("inputs")
        do_sample = get_typed_arg("do_sample", int, False)
        temperature = get_typed_arg("temperature", float)
        top_k = get_typed_arg("top_k", int)
        top_p = get_typed_arg("top_p", float)
        repetition_penalty = get_typed_arg("repetition_penalty", float)
        max_length = get_typed_arg("max_length", int)
        max_new_tokens = get_typed_arg("max_new_tokens", int)
        logger.info(f"generate(), {model_name=}, {inputs=}")

        model, tokenizer, backend_config = models[model_name]
        if not backend_config.public_api:
            raise ValueError(f"We do not provide public API for {model_name} due to license restrictions")

        if inputs is not None:
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            n_input_tokens = inputs.shape[1]
        else:
            n_input_tokens = 0

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
        logger.info(f"generate(), outputs={repr(outputs)}")

        return jsonify(ok=True, outputs=outputs)
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
            if file_extension == ".txt" or file_extension == ".clean":
                loader = loader_class(file_path,autodetect_encoding=True)
            else:
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

def remove_patterns_from_text(text: str, start_pattern: str) -> str:
    answer_start = text.find(start_pattern)
    cleaned_output = text
    if answer_start != -1:
        cleaned_output = text[answer_start + len(start_pattern):].strip()
    else:
        cleaned_output = text.strip()  # In case "Answer:" is not found
    
    cleaned_output = cleaned_output.strip('"')
    return cleaned_output