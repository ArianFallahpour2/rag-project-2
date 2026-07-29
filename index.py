import json
import os
import streamlit as st
from groq import Groq
from google import genai

# Import retrieval functions (ensure these use API embeddings, not local PyTorch)
import functions.retrieve as rtrve
import functions.retrieve_childs as retrieve_childs

# Fetch API keys from Streamlit Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def call_groq(prompt, model_name):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in Streamlit secrets")
    
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content


def call_gemini(prompt):
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not set in Streamlit secrets")
    
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text


def rag_chat(query, history, selected_mode, show_sources):
    # 1. Load data files
    with open("child_chunks.json", "r", encoding="utf-8") as f:
        child_chunks = json.load(f)
    with open("parent_list.json", "r", encoding="utf-8") as f:
        parent_list = json.load(f)

    # 2. Retrieve context
    if selected_mode == "بازیابی قطعات فرزند":
        retrieved_texts = retrieve_childs.child_retriever(query, child_chunks)
    else:
        retrieved_texts = rtrve.retriever(query, child_chunks, parent_list)

    if not retrieved_texts:
        return "اطلاعات کافی در مورد سوال پرسیده شده موجود نیست."

    # 3. Build the prompt
    context = "\n\n".join(retrieved_texts)
    prompt = f"""شما یک دستیار دوستانه هستید.
اگر متن‌های مرجع زیر حاوی پاسخ سوال هستند، فقط بر اساس آنها پاسخ دهید.
اگر متن‌ها خالی یا نامرتبط هستند، می‌توانید یک پاسخ کوتاه و طبیعی بدهید (مثلاً به احوالپرسی جواب دهید)، ولی هرگز اطلاعات غیرمستند را به عنوان واقعیت بیان نکنید.

متن: {context}

سوال: {query}
"""

    # 4. Try Groq models first; fall back to Gemini
    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]
    
    answer = None
    for model_name in models:
        try:
            answer = call_groq(prompt, model_name)
            break
        except Exception as e:
            print(f"Groq {model_name} failed: {e}")

    if answer is None:
        try:
            answer = call_gemini(prompt)
        except Exception as gemini_error:
            print(f"Gemini failed: {gemini_error}")
            return "متأسفانه در حال حاضر سرویس پاسخ‌گویی در دسترس نیست."

    # 5. Optionally attach source references
    if show_sources:
        header = "\n\n📚 **منابع استفاده شده:**\n"
        sources = [f"**{i+1}.** {chunk}" for i, chunk in enumerate(retrieved_texts)]
        answer += header + "\n\n".join(sources)

    return answer


def rag_chat_simple(message):
    return rag_chat(message, [], "بازیابی قطعات فرزند", False)
