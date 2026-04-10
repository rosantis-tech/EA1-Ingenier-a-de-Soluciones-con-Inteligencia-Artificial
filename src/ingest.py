"""
Módulo de ingesta de documentos para el pipeline RAG de BancoEstado.

Responsabilidades:
- Cargar documentos PDF, DOCX y TXT desde el directorio de datos.
- Dividir los documentos en chunks con RecursiveCharacterTextSplitter.
- Generar embeddings con OpenAI y almacenarlos en FAISS.
"""

import os
import glob

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
)


def load_documents(data_dir: str = DATA_DIR) -> list:
    """Carga todos los PDF, DOCX y TXT del directorio de datos."""
    documents = []

    pdf_files = glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    docx_files = glob.glob(os.path.join(data_dir, "**/*.docx"), recursive=True)
    for docx_path in docx_files:
        loader = Docx2txtLoader(docx_path)
        documents.extend(loader.load())

    txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    for txt_path in txt_files:
        loader = TextLoader(txt_path, encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(
            f"No se encontraron documentos en '{data_dir}'. "
            "Coloque los archivos normativos en esa carpeta."
        )

    print(f"[Ingesta] {len(documents)} documento(s) cargados desde '{data_dir}'.")
    return documents


def split_documents(documents: list) -> list:
    """Divide documentos en chunks usando RecursiveCharacterTextSplitter.

    Estrategia de chunking:
    - chunk_size=1000: suficiente para capturar un artículo normativo completo.
    - chunk_overlap=200: preserva contexto entre chunks adyacentes,
      crítico para normativa donde una oración puede depender de la anterior.
    - Separadores jerárquicos: primero por secciones, luego párrafos,
      luego oraciones, y finalmente por espacio.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingesta] {len(chunks)} chunks generados (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def create_vector_store(chunks: list) -> FAISS:
    """Genera embeddings y almacena en FAISS con persistencia local."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"[Ingesta] Vector store creado en '{FAISS_INDEX_DIR}' con {len(chunks)} vectores.")
    return vector_store


def run_ingestion() -> FAISS:
    """Ejecuta el pipeline completo de ingesta."""
    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    return vector_store


if __name__ == "__main__":
    run_ingestion()
    print("[Ingesta] Proceso completado exitosamente.")
