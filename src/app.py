"""
Interfaz Streamlit para el Asistente Normativo de BancoEstado.

Permite a los funcionarios realizar consultas sobre normativas
a través de una interfaz conversacional web.
"""

import streamlit as st
from rag_chain import query, get_vector_store, build_rag_chain

st.set_page_config(
    page_title="Asistente Normativo - BancoEstado",
    page_icon="🏦",
    layout="wide",
)

st.title("Asistente Normativo BancoEstado")
st.caption("Consulta normativas CMF y políticas internas de forma rápida y confiable.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Verificar que el vector store existe
try:
    get_vector_store()
    vs_ready = True
except FileNotFoundError:
    vs_ready = False

if not vs_ready:
    st.warning(
        "No se ha encontrado la base de datos vectorial. "
        "Ejecute primero la ingesta de documentos:\n\n"
        "```bash\npython -m src.ingest\n```"
    )
    st.stop()

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Ver fuentes consultadas"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** (pág. {s['page']})")

# Input del usuario
if prompt := st.chat_input("Escriba su consulta normativa..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando normativas..."):
            result = query(prompt)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("Ver fuentes consultadas"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}** (pág. {s['page']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
