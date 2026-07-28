import os
import zipfile
import json
import pandas as pd
from datetime import datetime
import chainlit as cl
from huggingface_hub import HfApi
import index

# ------------------------------------------------------------
# Extract data files from zip (runs once on application boot)
# ------------------------------------------------------------
ZIP_FILENAME = "data.zip"

if not os.path.exists("child_index.faiss"):
    if os.path.exists(ZIP_FILENAME):
        print(f"Extracting {ZIP_FILENAME}...")
        with zipfile.ZipFile(ZIP_FILENAME, "r") as zf:
            zf.extractall(".")
        print("Extraction complete.")
    else:
        print(f"WARNING: {ZIP_FILENAME} not found! The app may not work.")

# ------------------------------------------------------------
# Hugging Face Dataset Feedback Configuration
# ------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "arn-flp/RAG-project-data"
FEEDBACK_FILE = "feedback.csv"


def push_feedback_file():
    """Upload the local feedback CSV to Hugging Face Dataset repository."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is not set. Feedback will not be uploaded.")
        return

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=FEEDBACK_FILE,
            path_in_repo=FEEDBACK_FILE,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Update feedback via Chainlit UI",
        )
        print("Successfully uploaded feedback.csv to Hugging Face Datasets.")
    except Exception as e:
        print(f"Error uploading feedback to Hugging Face: {e}")


def save_feedback(query: str, answer: str, rating: str):
    """Save feedback locally to CSV and trigger Hugging Face upload."""
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
        
    print(f"Saved feedback to {os.path.abspath(FEEDBACK_FILE)} | Rating: {rating}")
    push_feedback_file()


# ------------------------------------------------------------
# Chainlit Lifecycle & Action Handlers
# ------------------------------------------------------------

@cl.action_callback("feedback_like")
async def handle_like(action: cl.Action):
    """Handles Positive Feedback (👍)"""
    query = action.payload.get("query", "")
    answer = action.payload.get("answer", "")
    
    save_feedback(query, answer, "like")
    
    await cl.Message(content="👍 متشکریم").send()
    await action.remove()


@cl.action_callback("feedback_dislike")
async def handle_dislike(action: cl.Action):
    """Handles Negative Feedback (👎)"""
    query = action.payload.get("query", "")
    answer = action.payload.get("answer", "")
    
    save_feedback(query, answer, "dislike")
    
    await cl.Message(content="👎 متشکریم، بررسی می‌کنیم").send()
    await action.remove()


@cl.on_chat_start
async def on_chat_start():
    """Triggered when a user connects or resets the chat session."""
    await cl.Message(
        content="سلام! سیستم پاسخگویی به سوالات (RAG) آماده است. لطفاً پرسش خود را وارد کنید."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Triggered when the user submits a message."""
    user_query = message.content

    # 1. Generate answer via index.py
    async with cl.Step(name="بازیابی و پردازش (RAG)"):
        answer = index.rag_chat_simple(user_query)

    # 2. Attach Feedback Action Buttons to the response
    actions = [
        cl.Action(
            name="feedback_like",
            label="👍",
            payload={"query": user_query, "answer": answer}
        ),
        cl.Action(
            name="feedback_dislike",
            label="👎",
            payload={"query": user_query, "answer": answer}
        )
    ]

    # 3. Send response with action buttons
    await cl.Message(content=answer, actions=actions).send()
