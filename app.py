import os
import zipfile

# ------------------------------------------------------------
# Extract data files from the zip (only once, before anything else)
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

from datetime import datetime

HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
import index
import pandas as pd

feedback_file = "feedback.csv"
HF_DATASET_REPO = "arn-flp/RAG-project-data"


def push_feedback_file():
  """Upload the local CSV to the dataset repo."""
  if not HF_TOKEN:
    print("WARNING: HF_TOKEN is not set. Feedback will not be uploaded.")
    return

  api = HfApi()
  api.upload_file(
      path_or_fileobj="feedback.csv",
      path_in_repo="feedback.csv",
      repo_id=HF_DATASET_REPO,
      repo_type="dataset",
      token=HF_TOKEN,
      commit_message="Update feedback",
  )


def save_feedback(query, answer, rating):
    # Download existing feedback if it exists
    try:
        local_file = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=feedback_file,
            token=HF_TOKEN,
            force_download=True,
        )
        df = pd.read_csv(local_file)
    except Exception:
        # Dataset is empty or file doesn't exist yet
        df = pd.DataFrame(columns=["timestamp", "query", "answer", "rating"])

    # Add new row
    df.loc[len(df)] = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "rating": rating,
    }

    # Save locally
    df.to_csv("feedback.csv", index=False)

    # Upload to HF
    push_feedback_file()


import chainlit as cl

# ----------------------------
# Chat starts
# ----------------------------
@cl.on_chat_start
async def start():
    await cl.Message(
        content="سلام! 👋\n\nسوال خود را بپرسید."
    ).send()


# ----------------------------
# User sends a message
# ----------------------------
@cl.on_message
async def on_message(message: cl.Message):

    query = message.content

    answer = index.rag_chat_simple(query)

    # Save for feedback later
    cl.user_session.set("last_query", query)
    cl.user_session.set("last_answer", answer)

    actions = [
        cl.Action(
            name="like",
            payload={"rating": "like"},
            label="👍"
        ),
        cl.Action(
            name="dislike",
            payload={"rating": "dislike"},
            label="👎"
        )
    ]

    await cl.Message(
        content=answer,
        actions=actions
    ).send()


# ----------------------------
# Like button
# ----------------------------
@cl.action_callback("like")
async def like(action: cl.Action):

    query = cl.user_session.get("last_query")
    answer = cl.user_session.get("last_answer")

    save_feedback(query, answer, "like")

    await action.remove()

    await cl.Message(
        content="👍 متشکریم"
    ).send()


# ----------------------------
# Dislike button
# ----------------------------
@cl.action_callback("dislike")
async def dislike(action: cl.Action):

    query = cl.user_session.get("last_query")
    answer = cl.user_session.get("last_answer")

    save_feedback(query, answer, "dislike")

    await action.remove()

    await cl.Message(
        content="👎 متشکریم، بررسی می‌کنیم"
    ).send()




# ---- Launch ----
if __name__ == "__main__":
    import os
    import sys
    from chainlit.cli import run_chainlit

    port = os.getenv("PORT", "7860")

    sys.argv = [
        "chainlit",
        "run",
        __file__,
        "--host",
        "0.0.0.0",
        "--port",
        port,
    ]

    run_chainlit(__file__)
