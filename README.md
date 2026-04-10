# Asistente Normativo BancoEstado - EA1 ISY0101

Solución basada en **agentes LLM** y **pipeline RAG** para interpretar y simplificar normativas de la CMF y políticas internas de BancoEstado, dirigida a ejecutivos de sucursal.

**Asignatura:** Ingeniería de Soluciones con Inteligencia Artificial (ISY0101)  
**Sección:** 008V | **Docente:** Javier Esteban Peña Reyes  
**Integrantes:** Roger Rosas Peña, Edgardo Gutierrez, Rodrigo Santis Erices

---

## Arquitectura

```
Funcionario ──► Streamlit UI ──► LangChain RetrievalQA
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                   ChromaDB        GPT-4o          Prompt
                 (Vector Store)    (LLM)         (System +
                        │                        User RAG)
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        OpenAI Embeddings    Documentos
      (text-embedding-3-small) (PDF/DOCX)
```

### Componentes

| Componente | Tecnología | Justificación |
|---|---|---|
| LLM | GPT-4o | Alta capacidad de razonamiento contextual y comprensión de lenguaje jurídico |
| Embeddings | text-embedding-3-small | Balance costo/rendimiento para búsqueda semántica |
| Vector Store | ChromaDB | Ligero, sin servidor externo, persistencia local |
| Framework | LangChain | Orquestación estándar de cadenas RAG |
| Chunking | RecursiveCharacterTextSplitter | Preserva estructura de artículos normativos |
| Interfaz | Streamlit | Prototipado rápido de interfaces conversacionales |

---

## Requisitos

- Python 3.10+
- API Key de OpenAI

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/rogrosas/EA1-Ingenier-a-de-Soluciones-con-Inteligencia-Artificial.git
cd EA1-Ingenier-a-de-Soluciones-con-Inteligencia-Artificial
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env y agregar su OPENAI_API_KEY
```

### 4. Agregar documentos normativos

Coloque los archivos PDF y/o DOCX de normativas en la carpeta `data/`:

```
data/
├── normativa_cmf_bancos.pdf
├── manual_cumplimiento_interno.pdf
└── politicas_operativas.docx
```

### 5. Ejecutar la ingesta de documentos

```bash
python -m src.ingest
```

Esto cargará los documentos, los dividirá en chunks, generará embeddings y los almacenará en ChromaDB.

### 6. Iniciar la aplicación

```bash
streamlit run src/app.py
```

La interfaz estará disponible en `http://localhost:8501`.

---

## Estructura del Proyecto

```
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuración centralizada y prompts
│   ├── ingest.py          # Pipeline de ingesta (load → chunk → embed → store)
│   ├── rag_chain.py       # Cadena RAG con LangChain (retriever + LLM)
│   └── app.py             # Interfaz Streamlit
├── data/                  # Documentos normativos (no versionados)
├── .env.example           # Template de variables de entorno
├── requirements.txt       # Dependencias Python
└── README.md
```

## Pipeline RAG - Flujo de Datos

1. **Ingesta:** Los documentos PDF/DOCX se cargan con `PyPDFLoader` y `Docx2txtLoader`.
2. **Chunking:** Se dividen con `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) usando separadores jerárquicos (`\n\n`, `\n`, `. `).
3. **Embedding:** Cada chunk se vectoriza con `text-embedding-3-small` de OpenAI.
4. **Almacenamiento:** Los vectores se persisten en ChromaDB local.
5. **Consulta:** La pregunta del usuario se vectoriza y se buscan los 4 chunks más similares (cosine similarity).
6. **Generación:** GPT-4o recibe el system prompt + contexto recuperado + pregunta, y genera una respuesta citando fuentes.

## Uso de IA en el Proyecto

Se utilizó **Claude** (Anthropic) como herramienta de apoyo para la estructuración del código y la redacción del informe técnico. Todas las decisiones de diseño, análisis del caso y justificaciones técnicas fueron elaboradas por el equipo.

## Referencias

- Normativa de Bancos e Instituciones Financieras - CMF Chile. Recuperado de https://www.cmfchile.cl/portal/principal/613/w3-propertyvalue-43581.html
- LangChain Documentation. https://python.langchain.com/
- ChromaDB Documentation. https://docs.trychroma.com/
- OpenAI API Reference. https://platform.openai.com/docs/
