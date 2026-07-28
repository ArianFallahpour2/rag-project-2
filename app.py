import os
import zipfile
from datetime import datetime
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi

# ------------------------------------------------------------
# 1. Page Configuration & RTL Custom CSS
# ------------------------------------------------------------
st.set_page_config(page_title="Farsi RAG", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    /* Global Right-to-Left alignment for Farsi UI */
    html, body, [data-testid="stAppViewContainer"] {
        direction: rtl;
        text-align: right;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    .stTextInput input, .stChatInput textarea {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="column"] {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 2. Extract data files from zip (only once)
# ------------------------------------------------------------
ZIP_FILENAME = "data.zip"

if not os.path.exists("child_index.faiss"):
    if os.path.exists(ZIP_FILENAME):
        st.write(f"Extracting {ZIP_FILENAME}...")
        with zipfile.ZipFile(ZIP_FILENAME, "r") as zf:
            zf.extractall(".")
        st.write("Extraction complete.")
    else:
        st.warning(f"WARNING: {ZIP_FILENAME} not found! The app may not work.")

# Import indexing logic after extraction
import index  # Ensure index.py imports cleanly

# ------------------------------------------------------------
# 3. Environment & Secrets Setup
# ------------------------------------------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else os.getenv("HF_TOKEN")
HF_DATASET_REPO = "arn-flp/RAG-project-data"
FEEDBACK_FILE = "feedback.csv"


def push_feedback_file():
    """Upload the local feedback CSV to the Hugging Face dataset repo."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is not set. Feedback will not be uploaded.")
        return

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=FEEDBACK_FILE,
            path_in_repo="feedback.csv",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Update feedback via Streamlit",
        )
    except Exception as e:
        print(f"Error uploading feedback to HF: {e}")


def save_feedback(query, answer, rating):
    df_new = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "rating": rating,
    }])
    if not os.path.exists(FEEDBACK_FILE):
        df_new.to_csv(FEEDBACK_FILE, index=False)
    else:
        df_new.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    
    push_feedback_file()


# ------------------------------------------------------------
# 4. Session State Initialization
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None  # Stores feedback state for the latest response

# ------------------------------------------------------------
# 5. UI Structure
# ------------------------------------------------------------
st.title("سامانه پرسش و پاسخ هوشمند (Farsi RAG)")

# Sidebar control to clear chat
with st.sidebar:
    if st.button("پاک کردن گفتگو", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_feedback = None
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("لطفا پرسش خود را وارد کنید"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("در حال جستجو و پاسخ‌دهی..."):
            answer = index.rag_chat_simple(user_query)
            st.markdown(answer)

    # Append assistant response & reset feedback state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_feedback = None
    st.rerun()

# ------------------------------------------------------------
# 6. Feedback Mechanism (Visible if at least one exchange exists)
# ------------------------------------------------------------
if st.session_state.messages:
    # Get last pair of user question and assistant answer
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]

    if user_msgs and assistant_msgs:
        last_q = user_msgs[-1]["content"]
        last_a = assistant_msgs[-1]["content"]

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            if st.button("👍", key="like_btn", disabled=(st.session_state.last_feedback is not None)):
                save_feedback(last_q, last_a, "like")
                st.session_state.last_feedback = "like"
                st.rerun()

        with col2:
            if st.button("👎", key="dislike_btn", disabled=(st.session_state.last_feedback is not None)):
                save_feedback(last_q, last_a, "dislike")
                st.session_state.last_feedback = "dislike"
                st.rerun()

        with col3:
            if st.session_state.last_feedback == "like":
                st.success("👍 متشکریم")
            elif st.session_state.last_feedback == "dislike":
                st.info("👎 متشکریم، بررسی می‌کنیم")
