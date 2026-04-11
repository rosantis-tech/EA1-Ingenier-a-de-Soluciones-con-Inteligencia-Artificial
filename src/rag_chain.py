"""
Módulo de cadena RAG para el asistente normativo de BancoEstado.

Responsabilidades:
- Cargar el vector store FAISS desde disco.
- Construir la cadena de RetrievalQA con LangChain.
- Exponer una función de consulta con trazabilidad de fuentes.
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from config import (
    GITHUB_TOKEN,
    GITHUB_BASE_URL,
    GITHUB_EMBEDDINGS_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    RETRIEVER_K,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


def get_vector_store() -> FAISS:
    """Carga el vector store existente desde disco."""
    if not os.path.exists(FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"No se encontró el vector store en '{FAISS_INDEX_DIR}'. "
            "Ejecute primero: python -m src.ingest"
        )
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=GITHUB_TOKEN,
        openai_api_base=GITHUB_EMBEDDINGS_URL,
    )
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_rag_chain(vector_store: FAISS) -> RetrievalQA:
    """Construye la cadena RAG con retriever y LLM."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=GITHUB_TOKEN,
        openai_api_base=GITHUB_BASE_URL,
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE),
    ])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def query(question: str) -> dict:
    """Realiza una consulta al pipeline RAG.

    Returns:
        dict con claves:
            - "answer": respuesta generada por el LLM.
            - "sources": lista de metadatos de los documentos fuente.
    """
    vector_store = get_vector_store()
    chain = build_rag_chain(vector_store)
    result = chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "source": doc.metadata.get("source", "Desconocido"),
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:200] + "...",
        })

    return {
        "answer": result["result"],
        "sources": sources,
    }
