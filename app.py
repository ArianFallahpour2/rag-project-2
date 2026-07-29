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
import gradio as gr

HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
import index
import pandas as pd

# ---- RTL CSS (applied via launch()) ----
css = """
  html, body, .gradio-container {
    direction: rtl;
  }
  .gradio-container .chatbot {
    direction: rtl;
  }
  .gradio-container .textbox textarea {
    direction: rtl;
    text-align: right;
  }
  .gradio-container .message {
    direction: rtl;
    unicode-bidi: isolate;
    text-align: right;
  }
"""

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


# ---- Build the Gradio interface ----
with gr.Blocks(title="Farsi RAG") as demo:
  chat_bot = gr.Chatbot()
  msg = gr.Textbox(placeholder="لطفا پرسش خود را وارد کنید")
  clear = gr.Button("پاک کردن گفتگو")

  # Feedback row
  with gr.Row(visible=False) as feedback_row:
    like_btn = gr.Button("👍")
    dislike_btn = gr.Button("👎")
    feedback_msg = gr.Markdown("")

  def get_message_text(msg):
    content = msg.get("content", "")
    if isinstance(content, str):
      return content
    if isinstance(content, list) and len(content) > 0:
      return content[0].get("text", "")
    return str(content)

  def respond(query, chat_history):
    answer = index.rag_chat_simple(query)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    return (
        "",
        chat_history,
        gr.update(visible=True),  # feedback_row
        gr.update(interactive=True),  # like_btn
        gr.update(interactive=True),  # dislike_btn
        "",  # feedback_msg
    )

  def handle_like(chat_history):
    if chat_history:
      user_msgs = [m for m in chat_history if m["role"] == "user"]
      assistant_msgs = [m for m in chat_history if m["role"] == "assistant"]
      if user_msgs and assistant_msgs:
        q = get_message_text(user_msgs[-1])
        a = get_message_text(assistant_msgs[-1])
        save_feedback(q, a, "like")
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            "👍 متشکریم",
        )
    return (gr.update(interactive=False), gr.update(interactive=False), "")

  def handle_dislike(chat_history):
    if chat_history:
      user_msgs = [m for m in chat_history if m["role"] == "user"]
      assistant_msgs = [m for m in chat_history if m["role"] == "assistant"]
      if user_msgs and assistant_msgs:
        q = get_message_text(user_msgs[-1])
        a = get_message_text(assistant_msgs[-1])
        save_feedback(q, a, "dislike")
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            "👎 متشکریم، بررسی می‌کنیم",
        )
    return (gr.update(interactive=False), gr.update(interactive=False), "")

  # Wire events
  msg.submit(
      respond,
      [msg, chat_bot],
      [msg, chat_bot, feedback_row, like_btn, dislike_btn, feedback_msg],
  )
  like_btn.click(
      handle_like, chat_bot, [like_btn, dislike_btn, feedback_msg]
  )
  dislike_btn.click(
      handle_dislike, chat_bot, [like_btn, dislike_btn, feedback_msg]
  )
  clear.click(
      lambda: (
          [],  # chat_bot
          gr.update(visible=False),  # feedback_row
          gr.update(interactive=True),  # like_btn
          gr.update(interactive=True),  # dislike_btn
          "",  # feedback_msg
      ),
      [],
      [chat_bot, feedback_row, like_btn, dislike_btn, feedback_msg],
  )

# ---- Launch ----
if __name__ == "__main__":
  port = int(os.getenv("PORT", 7860))
  demo.launch(server_name="0.0.0.0", server_port=port, css=css)
