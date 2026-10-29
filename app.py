# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import asyncio
import tempfile
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import time
from langchain_community.document_loaders import PyPDFLoader

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/gemini-embedding-001"
RETRIEVER_K = 4
DEFAULT_SYSTEM_MESSAGE = """
You are Document RAG Assistant 📄🤖. 
Your role is to help users understand and explore the content of uploaded documents.

Follow these rules:
1. Always prioritize the document context when answering questions.
2. If the answer is not in the document, clearly say you don't know.
3. Keep responses friendly, clear, and concise.
"""

# Load environment variables
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
    # if "current_video_id" not in st.session_state:
    # st.session_state.current_video_id = None


def configure_page():
    st.set_page_config(
        page_title="Document RAG Chat",
        page_icon="📄",
        layout="centered",
    )

    st.title("📄 Document RAG Assistant")
    st.markdown("### Chat with your documents using AI")


def center_app():
    st.markdown(
        """
        <style>
        /* Center the main content */
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            # text-align: center;
            max-width: 800px; /* prevent content from stretching too wide */
            margin: auto;
        }

        #transform-any-you-tube-video-into-an-interactive-conversation,#you-tube-x-rag-assistant{
        text-align:center;
        }

        /* Center text inputs and buttons */
        .stTextInput, .stButton {
            width: 100% !important;
            # max-width: 500px;
            margin: auto;
        }

        @media (max-width: 768px) {
                .stVerticalBlock{
                align-items:center;
                }
            }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def handle_new_document_button():
    if st.sidebar.button("🔄 New Document", use_container_width=True):
        # Clear document-related session state
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        if "document_name" in st.session_state:
            del st.session_state["document_name"]

        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
        st.success("🔄 Ready for new document!")
        time.sleep(1)
        st.rerun()


def handle_sidebar():
    # Sidebar for API key
    st.sidebar.header("🔑 Configuration")

    api_key = st.sidebar.text_input(
        "Your Google Gemini API Key",
        type="password",
        placeholder="Enter your API key...",
        help="Your key is kept only in your current browser session.",
        value=st.session_state.get("api_key", ""),
    )
    if api_key:
        clean_key = api_key.strip()
        st.session_state.api_key = clean_key
        os.environ["GOOGLE_API_KEY"] = clean_key
        if len(clean_key) < 15:
            st.sidebar.error("⚠️ This API key looks too short. Please check it.")
        else:
            st.sidebar.success("✅ API key set for this session")
    else:
        st.sidebar.info("💡 Enter your API key to start chatting")

    st.sidebar.divider()

    selected_model = st.sidebar.selectbox(
        "Generation Models",
        [
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-flash-latest",
            "gemini-pro-latest",
        ],
        index=0,
        help="Choose the Gemini model for generation",
    )

    st.session_state.model = selected_model

    st.sidebar.divider()

    st.sidebar.subheader("💬 Chat Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()

    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    handle_new_document_button()

    st.sidebar.divider()
    st.sidebar.subheader("📊 Session Info")

    message_count = len(st.session_state.messages) - 1  # Exclude system message
    document_processed = (
        "retriever" in st.session_state
        and st.session_state.get("retriever") is not None
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Messages", message_count)
    with col2:
        st.metric("Document", "✅" if document_processed else "❌")

    if document_processed:
        st.sidebar.success("📄  Document ready for chat")
    else:
        st.sidebar.info("📄  No Document processed yet")

    st.sidebar.info(f"**Current Model:**\n{selected_model}")

    if message_count > 0:
        st.sidebar.divider()
        chat_text = ""
        for msg in st.session_state.messages[1:]:  # Skip system message
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_text += f"{role}: {msg.content}\n\n"

        st.sidebar.download_button(
            "📥 Download Chat",
            chat_text,
            f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True,
            help="Download your conversation history",
        )

    # Main interface
    uploaded_file = st.file_uploader(
        "📁 Upload your document",
        type=["pdf", "txt"],
        help="Upload a PDF or text file to chat with",
    )

    return selected_model, uploaded_file, st.session_state.get("api_key")


def handle_document_processing(uploaded_file=""):
    if st.button("🚀 Process Document", type="primary"):
        user_api_key = st.session_state.get("api_key", "") or os.getenv("GOOGLE_API_KEY", "")
        if not user_api_key:
            st.error("❌ Please enter your Google Gemini API key in the sidebar first!")
            st.info("💡 You need a valid API key to process documents and chat")
            return
        elif not uploaded_file:
            st.error("❌ Please upload a document first!")
            return
        else:
            with st.spinner("Processing document..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:

                    # Step 1: Extract transcript
                    status_text.text("🔄 Step 1/4: Saving document...")
                    progress_bar.progress(25)

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    status_text.text("📄 Step 2/4: Loading document...")
                    progress_bar.progress(50)

                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file_path)
                    else:  # txt file
                        loader = TextLoader(tmp_file_path)

                    documents = loader.load()

                    status_text.text("✂️ Step 3/4: Splitting into chunks...")
                    progress_bar.progress(75)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    chunks = splitter.split_documents(documents)

                    status_text.text("🧠 Step 4/4: Creating embeddings...")
                    progress_bar.progress(100)
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model=EMBEDDING_MODEL,
                        google_api_key=user_api_key,
                    )
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
                    )

                    st.session_state["retriever"] = retriever
                    st.session_state["document_name"] = uploaded_file.name

                    os.unlink(tmp_file_path)
                    progress_bar.empty()
                    status_text.empty()

                    st.success("✅ Document processed! Ready for questions.")
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


@st.cache_resource()
def get_chat_model(model_name: str, api_key_keyed_for_cache: str | None):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key_keyed_for_cache,
    )


def display_chat_messages():
    for message in st.session_state.messages[1:]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)


def handle_user_input(chat_model, input_disabled: bool = False):
    if prompt := st.chat_input(
        "Ask a question about the document...", disabled=input_disabled
    ):
        if not prompt.strip():
            st.warning("Please type a message before sending!")
            return

        st.session_state.messages.append(HumanMessage(content=prompt))

        prompt_template = PromptTemplate(
            template="""Based on this document content:

            {context}

            Question: {question}""",
            input_variables=["context", "question"],
        )

        with st.chat_message("user"):
            st.write(prompt)

        retriever = st.session_state.get("retriever")
        if not retriever:
            with st.chat_message("assistant"):
                error_msg = (
                    "❌ Please process a document first to enable question answering."
                )
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
            return
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing document content..."):
                try:
                    retrieved_docs = retriever.invoke(prompt)
                    if not retrieved_docs:
                        no_context_msg = "🤷‍♂️ I couldn't find relevant information in the document for your question."
                        st.warning(no_context_msg)
                        st.session_state.messages.append(
                            AIMessage(content=no_context_msg)
                        )
                        return
                    parallel_chain = RunnableParallel(
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough(),
                        }
                    )
                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt_template | chat_model | parser

                    message_placeholder = st.empty()
                    full_response = ""

                    # Stream the response using stream method (synchronous)
                    for chunk in main_chain.stream(prompt):
                        if chunk and chunk.strip():
                            full_response += chunk
                            message_placeholder.markdown(
                                full_response + "▌"
                            )  # Cursor indicator

                    # Remove cursor and display final response
                    if full_response and full_response.strip():
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            AIMessage(content=full_response)
                        )
                    else:
                        error_msg = (
                            "🚫 No response received. Please try a different model."
                        )
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append(AIMessage(content=error_msg))

                    # Rerun to refresh the UI after streaming
                    st.rerun()

                except Exception as e:
                    error_message = str(e).lower()
                    if "not found" in error_message or "invalid" in error_message:
                        error_msg = "❌ This model is not available. Please select a different model."
                    elif "quota" in error_message or "limit" in error_message:
                        error_msg = "📊 API quota exceeded. Please try again later or use a different model."
                    elif "timeout" in error_message:
                        error_msg = (
                            "⏱️ Request timed out. Try a different model or try again."
                        )
                    else:
                        error_msg = f"❌ An error occurred. Try selecting different model or check your api key:("

                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
            # st.rerun()


init_session_state()
configure_page()
center_app()
selected_model, uploaded_file, user_api_key = handle_sidebar()
if uploaded_file:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.info(f"📊 Size: {uploaded_file.size} bytes")

handle_document_processing(uploaded_file)
chat_model = None
if user_api_key:
    # Ensure env var is set for the underlying client
    os.environ["GOOGLE_API_KEY"] = user_api_key
    chat_model = get_chat_model(selected_model, user_api_key)


display_chat_messages()

if chat_model is None:
    st.warning(
        "Please enter your Google Gemini API key in the sidebar to start chatting."
    )

handle_user_input(chat_model, input_disabled=(chat_model is None))
