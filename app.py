"""
College FAQ Assistant
=============================================================================
A Retrieval-Augmented Generation (RAG) chatbot designed for college websites.
Answers student, applicant, and parent questions based strictly on official
college documents (PDF/TXT) using LangChain, Google Gemini, and FAISS.
=============================================================================
"""

import os
import time
import asyncio
import tempfile
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# LangChain components for the RAG architecture
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Project Constants & Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000            # Size of each text chunk (characters)
CHUNK_OVERLAP = 100          # Overlap between adjacent chunks for context continuity
RETRIEVER_K = 4              # Number of top relevant chunks to retrieve
EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_SAMPLE_DOC = os.path.join(os.path.dirname(__file__), "data", "college_faq_handbook.txt")

# Grounded College FAQ Prompt Template to prevent hallucinations
COLLEGE_QA_PROMPT = PromptTemplate(
    template="""You are the official College FAQ Assistant for students, applicants, and parents.
Answer the student's question strictly using ONLY the factual information found in the College Document Context below.

College Document Context:
\"\"\"
{context}
\"\"\"

Student Question: {question}

Strict Instructions:
1. Answer ONLY using the facts stated in the context above.
2. If the context does not contain the answer, reply EXACTLY with:
   "I couldn't find this information in the available college documents."
3. Do NOT make up, assume, or infer any dates, numbers, fees, or policies that are not explicitly stated.
4. Keep answers friendly, accurate, and structured (use bullet points for lists like fees, eligibility criteria, or required documents).

Answer:""",
    input_variables=["context", "question"],
)

SUGGESTED_QUESTIONS = [
    "What is the admission process?",
    "What courses are available?",
    "What is the fee structure?",
    "What is the eligibility criteria?",
    "What documents are required for admission?",
    "What are the placement statistics?",
    "When do examinations start?",
    "What are the college timings?",
]

# Load environment variables (.env file)
load_dotenv()

# Setup event loop for async LangChain operations if not present
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
def init_session_state():
    """Initializes persistent session state variables for chat and documents."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Welcome to the **College FAQ Assistant**! "
                    "I can answer your questions about admissions, courses, fees, "
                    "examinations, placements, and campus life using official college documents.\n\n"
                    "👉 Load the sample handbook or upload your own college PDF/TXT to get started!"
                ),
                "sources": [],
            }
        ]
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "model" not in st.session_state:
        st.session_state.model = "gemini-3.5-flash"


# ---------------------------------------------------------------------------
# UI Page Setup & Styling
# ---------------------------------------------------------------------------
def configure_page():
    st.set_page_config(
        page_title="College FAQ Assistant",
        page_icon="🎓",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 860px;
            padding-top: 3.5rem;
            padding-bottom: 3rem;
            margin: auto;
        }
        .campus-badge-container {
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .campus-badge {
            display: inline-block;
            background-color: #4F46E5;
            color: #FFFFFF !important;
            padding: 0.35rem 0.95rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4);
        }
        h1 {
            text-align: center !important;
            padding-top: 0 !important;
            margin-bottom: 0.3rem !important;
        }
        .header-subtitle {
            text-align: center;
            font-size: 1.05rem;
            opacity: 0.85;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            border-radius: 8px;
            transition: all 0.2s ease-in-out;
        }
        </style>

        <div class="campus-badge-container">
            <span class="campus-badge">Campus AI Services</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎓 College FAQ Assistant")
    st.markdown(
        '<p class="header-subtitle">Ask questions about admissions, courses, fees, examinations, placements and other college information.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar Configuration & Controls
# ---------------------------------------------------------------------------
def handle_sidebar():
    st.sidebar.header("🎓 Portal Configuration")

    # API Key Input
    api_key_input = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Enter your Gemini API key...",
        help="Reads from .env by default. Enter here to override for this session.",
        value=st.session_state.get("api_key", ""),
    )
    if api_key_input:
        clean_key = api_key_input.strip()
        st.session_state.api_key = clean_key
        os.environ["GOOGLE_API_KEY"] = clean_key
        if len(clean_key) < 15:
            st.sidebar.error("⚠️ API key looks too short.")
        else:
            st.sidebar.success("✅ API key active")
    else:
        st.sidebar.info("💡 Enter your Gemini API key to chat")

    st.sidebar.divider()

    # Model Selection
    selected_model = st.sidebar.selectbox(
        "Gemini Generation Model",
        [
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-flash-latest",
            "gemini-pro-latest",
        ],
        index=0,
        help="Select the Gemini model for generating student answers.",
    )
    st.session_state.model = selected_model

    st.sidebar.divider()
    st.sidebar.subheader("📊 Knowledge Base Status")
    doc_loaded = st.session_state.get("retriever") is not None
    current_doc = st.session_state.get("document_name", "None")

    if doc_loaded:
        st.sidebar.success(f"📄 **Active:** `{current_doc}`")
    else:
        st.sidebar.warning("⚠️ No document indexed yet")

    st.sidebar.divider()
    st.sidebar.subheader("💬 Chat Controls")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I assist you with college information?",
                    "sources": [],
                }
            ]
            st.rerun()

    if doc_loaded:
        if st.sidebar.button("🔄 Unload Document", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.document_name = None
            st.success("Document unloaded!")
            time.sleep(0.5)
            st.rerun()

    # Download chat transcript
    if len(st.session_state.messages) > 1:
        transcript = ""
        for msg in st.session_state.messages:
            role = "Student" if msg["role"] == "user" else "Assistant"
            transcript += f"[{role}]: {msg['content']}\n\n"
        st.sidebar.download_button(
            "📥 Export Chat Transcript",
            transcript,
            file_name=f"college_faq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    return selected_model, st.session_state.get("api_key")


# ---------------------------------------------------------------------------
# Document Ingestion & RAG Indexing Pipeline
# ---------------------------------------------------------------------------
def process_document_to_retriever(file_path: str, doc_display_name: str, api_key: str):
    """
    RAG Pipeline Core:
    1. Document Loading (PDF / TXT)
    2. Recursive Text Chunking
    3. Gemini Vector Embeddings
    4. FAISS Vector Store Indexing
    5. Retriever Creation
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    # Tag document metadata with friendly display name for source attribution
    for doc in documents:
        doc.metadata["display_name"] = doc_display_name

    # Step 2: Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)

    # Step 3 & 4: Embeddings and FAISS Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 5: Similarity Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    st.session_state.retriever = retriever
    st.session_state.document_name = doc_display_name


def render_document_section(user_api_key: str):
    """Renders the document management section (upload or pre-loaded handbook)."""
    with st.expander("📁 Manage College Documents (Upload PDF/TXT or Load Sample)", expanded=(st.session_state.retriever is None)):
        tab1, tab2 = st.tabs(["📤 Upload College Document", "🏫 Load Sample College Handbook"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload official college PDF or TXT file (Prospectus, Syllabus, Rules, etc.)",
                type=["pdf", "txt"],
                help="Supported formats: PDF, TXT",
            )
            if uploaded_file:
                st.caption(f"Selected file: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
                if st.button("🚀 Process & Index Uploaded Document", type="primary", use_container_width=True):
                    if not user_api_key:
                        st.error("❌ Please provide a Google Gemini API Key in the sidebar first!")
                        return

                    with st.spinner("Processing document through RAG pipeline (Loading → Chunking → Embedding → FAISS)..."):
                        suffix = f".{uploaded_file.name.split('.')[-1]}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        try:
                            process_document_to_retriever(tmp_path, uploaded_file.name, user_api_key)
                            os.unlink(tmp_path)
                            st.success(f"✅ Successfully indexed **{uploaded_file.name}**! Ready for questions.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                            st.error(f"❌ Error processing document: {str(e)}")

        with tab2:
            st.markdown(
                "Quickly test the chatbot using the included **Apex Institute of Technology & Science (AITS) Official Student Handbook & FAQ (2025-2026)**."
            )
            st.markdown(
                "Covers: Admissions, B.Tech/BCA/MBA/MCA courses, fee structures, eligibility cutoffs, exam dates, 92% placement statistics, timings, and campus facilities."
            )
            if st.button("📚 Load Preloaded College Handbook", use_container_width=True):
                if not user_api_key:
                    st.error("❌ Please provide a Google Gemini API Key in the sidebar first!")
                    return
                if not os.path.exists(DEFAULT_SAMPLE_DOC):
                    st.error(f"❌ Sample file not found at {DEFAULT_SAMPLE_DOC}")
                    return

                with st.spinner("Indexing preloaded College Handbook with FAISS..."):
                    try:
                        process_document_to_retriever(
                            DEFAULT_SAMPLE_DOC,
                            "college_faq_handbook.txt",
                            user_api_key,
                        )
                        st.success("✅ Preloaded College Handbook indexed successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error indexing handbook: {str(e)}")


# ---------------------------------------------------------------------------
# Suggested Questions Component
# ---------------------------------------------------------------------------
def render_suggested_questions():
    """Displays one-click clickable questions for students."""
    st.markdown("##### 💡 Suggested Questions")
    cols = st.columns(2)
    for idx, question in enumerate(SUGGESTED_QUESTIONS):
        col = cols[idx % 2]
        if col.button(f"❓ {question}", key=f"sug_btn_{idx}", use_container_width=True):
            st.session_state.pending_prompt = question
            st.rerun()


# ---------------------------------------------------------------------------
# Chat Display & Answering Engine
# ---------------------------------------------------------------------------
def display_chat_messages():
    """Renders all conversation messages with student/assistant avatars and citations."""
    for message in st.session_state.messages:
        role = message["role"]
        avatar = "🧑‍🎓" if role == "user" else "🎓"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

            # Render document citation & excerpts if available
            sources = message.get("sources", [])
            if sources:
                with st.expander("📄 Referenced Document Sources", expanded=False):
                    for src in sources:
                        page_info = f" ({src['page']})" if src.get("page") else ""
                        st.markdown(f"• **Source:** `{src['filename']}`{page_info}")
                        st.caption(f"Excerpt: *\"{src['excerpt']}\"*")


def handle_question(prompt_text: str, model_name: str, user_api_key: str):
    """
    Executes the Question-Answering RAG flow:
    1. Append student question to UI.
    2. Query FAISS retriever for similarity chunks.
    3. Format context and prompt.
    4. Call Gemini LLM and stream response live.
    5. Append assistant answer with source citations.
    """
    # 1. Append student message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_text,
        "sources": [],
    })
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt_text)

    # Validate retriever
    retriever = st.session_state.get("retriever")
    if not retriever:
        no_doc_msg = (
            "⚠️ **No college document loaded!**\n\n"
            "Please upload a document or click **'Load Sample College Handbook'** above "
            "so I can search for accurate answers."
        )
        with st.chat_message("assistant", avatar="🎓"):
            st.warning(no_doc_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": no_doc_msg,
            "sources": [],
        })
        return

    # 2. Retrieve relevant chunks
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("🔍 Searching college documents..."):
            try:
                retrieved_docs = retriever.invoke(prompt_text)
            except Exception as e:
                st.error(f"❌ Error during retrieval: {str(e)}")
                return

        if not retrieved_docs:
            fallback_reply = "I couldn't find this information in the available college documents."
            st.markdown(fallback_reply)
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback_reply,
                "sources": [],
            })
            return

        # Prepare context and extract source citations
        context_parts = []
        sources = []
        doc_display = st.session_state.get("document_name", "College Document")

        for d in retrieved_docs:
            context_parts.append(d.page_content)
            page_num = d.metadata.get("page")
            page_label = f"Page {page_num + 1}" if page_num is not None else None
            filename = d.metadata.get("display_name", doc_display)
            sources.append({
                "filename": filename,
                "page": page_label,
                "excerpt": d.page_content.strip()[:240] + ("..." if len(d.page_content.strip()) > 240 else ""),
            })

        formatted_context = "\n\n---\n\n".join(context_parts)

        # 3. Build LangChain Runnable Chain
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=user_api_key,
            temperature=0.2,  # Low temperature for strict factual grounding
        )
        chain = COLLEGE_QA_PROMPT | llm | StrOutputParser()

        # 4. Stream response
        placeholder = st.empty()
        full_response = ""
        try:
            for chunk in chain.stream({"context": formatted_context, "question": prompt_text}):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"❌ Error communicating with Gemini API: {str(e)}"
            placeholder.error(full_response)

        # 5. Show source citations
        if sources:
            with st.expander("📄 Referenced Document Sources", expanded=False):
                for src in sources:
                    page_info = f" ({src['page']})" if src.get("page") else ""
                    st.markdown(f"• **Source:** `{src['filename']}`{page_info}")
                    st.caption(f"Excerpt: *\"{src['excerpt']}\"*")

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })


# ---------------------------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    configure_page()
    selected_model, user_api_key = handle_sidebar()

    # Document management section
    render_document_section(user_api_key)

    st.divider()

    # Display suggested questions
    render_suggested_questions()

    st.divider()

    # Display chat conversation
    display_chat_messages()

    # Determine input prompt (either from button click or chat input)
    prompt = None
    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        chat_box = st.chat_input("Ask a question about the college (admissions, fees, courses, placements...)...")
        if chat_box and chat_box.strip():
            prompt = chat_box.strip()

    if prompt:
        handle_question(prompt, selected_model, user_api_key)


if __name__ == "__main__":
    main()
