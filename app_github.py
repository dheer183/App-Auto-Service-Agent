# app.py - Final Version with Manual Inputs & Structured Output

import os
import zipfile
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from getpass import getpass
# Override system sqlite3 with modern version
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Fix event loop before Streamlit init
import asyncio
import nest_asyncio
nest_asyncio.apply()

# --------------------------
# INITIALIZATION
# --------------------------


@st.cache_resource
def initialize_system():
    """Initialize system components with SQLite3 workaround"""
    system = {}

    # Initialize event loop early
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Secure API handling
    system["groq_api_key"] = os.environ.get("GROQ_API_KEY", "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")  # Use st.secrets in prod

    # Initialize embeddings
    system["embeddings"] = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}  # Remove if using GPU
    )

    # Initialize vector databases
    system["retrievers"] = []
    vector_db_dir = "vectorDB/vectorDB/"

    if os.path.exists(vector_db_dir):
        for db_folder in os.listdir(vector_db_dir):
            db_path = os.path.join(vector_db_dir, db_folder)
            if os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
                try:
                    vectordb = Chroma(
                        persist_directory=db_path,
                        embedding_function=system["embeddings"]
                    )
                    system["retrievers"].append(vectordb.as_retriever())
                except Exception as e:
                    st.error(f"Failed loading {db_folder}: {str(e)}")
                    continue
            else:
                st.warning(f"Missing Chroma DB in: {db_path}")

    # Initialize LLM with error handling
    try:
        system["llm"] = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            api_key=system["groq_api_key"]
        )
        # ADD THIS SECTION FOR QA CHAIN
        system["qa_chain"] = load_qa_chain(
            system["llm"],
            chain_type="map_reduce"
        )
    except Exception as e:
        st.error(f"LLM initialization failed: {str(e)}")
        system["llm"] = None
        system["qa_chain"] = None  # Add fallback

    return system
# --------------------------
# CHATBOT PROMPT TEMPLATE
# --------------------------

CHATBOT_PROMPT = """
Automotive Assistant Protocol

**Vehicle Identification**
Make: {make}
Model: {model}
Year: {year}
Engine Size: {engine}

**Response Template**
**Service Estimate for {make} {model}**
| Component | Details | Price |
|-----------|---------|-------|
| Part | [Brand/Part Name] | $X.XX |
| Labor | [X] hours * 130 | $Y.YY |
| Total | | $Z.ZZ |

**Includes:**
- [Item 1]
- [Item 2]

**Recommended Services:**
| Service | Price |
|---------|-------|
| [Service 1] | $X.XX |

**Notes:**
- Mid-range parts selected for quality/value balance
- Professional installation recommended for complex repairs
- All prices in CAD

**Current Query:**
{user_input}

**Conversation History:**
{chat_history}
"""

# --------------------------
# MAIN APPLICATION
# --------------------------

def main():
    # Must be first Streamlit command
    st.set_page_config(
        page_title="Auto Service Advisor",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize components
    system = initialize_system()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vehicle_info" not in st.session_state:
        st.session_state.vehicle_info = {
            "make": "", "model": "", "year": "", "engine": ""
        }

    # Simple vehicle input form
    with st.expander("🚗 Enter Vehicle Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.vehicle_info["make"] = st.text_input("Make (e.g., Toyota)", value=st.session_state.vehicle_info["make"])
            st.session_state.vehicle_info["year"] = st.number_input("Year", min_value=1999, max_value=2001, step=1, value=2000 if not st.session_state.vehicle_info["year"] else st.session_state.vehicle_info["year"])
        with col2:
            st.session_state.vehicle_info["model"] = st.text_input("Model (e.g., Camry)", value=st.session_state.vehicle_info["model"])
            st.session_state.vehicle_info["engine"] = st.text_input("Engine Size (e.g., 2.5L)", value=st.session_state.vehicle_info["engine"])

    # Main chat interface
    st.title("Auto Service Assistant")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input
    if prompt := st.chat_input("Describe your vehicle issue"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Verify vehicle info
            vehicle_info = st.session_state.vehicle_info
            if not all(vehicle_info.values()):
                missing = [k for k,v in vehicle_info.items() if not v]
                st.error(f"Missing vehicle info: {', '.join(missing)}")
                return

            # Build chat history
            chat_history = "\n".join(
                [f"{msg['role']}: {msg['content']}"
                 for msg in st.session_state.messages]
            )

            # Format full prompt
            full_prompt = CHATBOT_PROMPT.format(
                chat_history=chat_history,
                user_input=prompt,
                **vehicle_info
            )

            # Retrieve relevant documents
            docs = []
            for retriever in system["retrievers"]:
                docs.extend(retriever.invoke(prompt))

            # Generate response
            with st.spinner("Analyzing your query..."):
                response = system["qa_chain"].invoke({
                    "input_documents": docs,
                    "question": full_prompt
                })

            # Display response
            with st.chat_message("assistant"):
                st.markdown(response["output_text"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["output_text"]
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
