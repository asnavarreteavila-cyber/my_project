import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from huggingface_hub import InferenceClient


MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CHATS_DIR = Path(__file__).with_name("chats")
MEMORY_PATH = Path(__file__).with_name("memory.json")
BASE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant in a Streamlit chat app. "
    "Answer clearly and keep track of the conversation context."
)


def load_hf_token():
    try:
        return str(st.secrets.get("HF_TOKEN", "")).strip()
    except Exception:
        return ""


def get_inference_client(token):
    return InferenceClient(api_key=token)


def load_memory():
    if not MEMORY_PATH.exists():
        return {}

    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    return data if isinstance(data, dict) else {}


def save_memory(memory):
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def clear_memory():
    save_memory({})


def merge_memory(existing_memory, new_memory):
    merged = dict(existing_memory)
    for key, value in new_memory.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue

        if isinstance(value, str):
            cleaned_value = value.strip()
            if cleaned_value:
                merged[normalized_key] = cleaned_value
        elif value not in (None, "", [], {}):
            merged[normalized_key] = value

    return merged


def chat_file_path(chat_id):
    return CHATS_DIR / f"{chat_id}.json"


def format_timestamp(timestamp_value):
    try:
        parsed = datetime.fromisoformat(timestamp_value)
        return parsed.strftime("%Y-%m-%d %I:%M %p")
    except ValueError:
        return timestamp_value


def load_saved_chats():
    CHATS_DIR.mkdir(exist_ok=True)
    chats = []
    for chat_file in sorted(CHATS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(chat_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, dict):
            continue

        chat_id = str(data.get("id", "")).strip()
        if not chat_id:
            continue

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        cleaned_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                cleaned_messages.append({"role": role, "content": content})

        created_at = str(data.get("created_at", "")).strip()
        if not created_at:
            created_at = datetime.now().isoformat()

        title = str(data.get("title", "New Chat")).strip() or "New Chat"
        chats.append(
            {
                "id": chat_id,
                "title": title,
                "created_at": created_at,
                "messages": cleaned_messages,
            }
        )

    chats.sort(key=lambda chat: chat["created_at"], reverse=True)
    return chats


def save_chat(chat):
    CHATS_DIR.mkdir(exist_ok=True)
    chat_file_path(chat["id"]).write_text(
        json.dumps(chat, indent=2),
        encoding="utf-8",
    )


def delete_chat_file(chat_id):
    file_path = chat_file_path(chat_id)
    if file_path.exists():
        file_path.unlink()


def build_system_prompt(memory):
    if not memory:
        return BASE_SYSTEM_PROMPT

    memory_summary = json.dumps(memory, ensure_ascii=True)
    return (
        f"{BASE_SYSTEM_PROMPT} "
        f"Here is saved user memory you should use for personalization when relevant: {memory_summary}"
    )


def build_model_messages(messages, memory):
    model_messages = [{"role": "system", "content": build_system_prompt(memory)}]
    model_messages.extend(messages)
    return model_messages


def request_json_completion(token, prompt):
    client = get_inference_client(token)
    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON with no extra explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
        )
    except Exception:
        return None

    choice = response.choices[0] if response and response.choices else None
    if not choice or not choice.message:
        return None

    return choice.message.content


def interpret_hf_error(error):
    error_text = str(error)
    lowered = error_text.lower()

    if "401" in lowered or "unauthorized" in lowered:
        return "Your Hugging Face token appears to be invalid."
    if "429" in lowered or "rate limit" in lowered:
        return "The Hugging Face API rate limit was reached. Please try again later."
    if "404" in lowered or "not found" in lowered:
        return "The selected Hugging Face model is not available on this inference backend."
    if "timeout" in lowered:
        return "The Hugging Face API timed out. Try again in a moment."

    return "A Hugging Face API error occurred. Please try again later."


def parse_json_object(text):
    if not text:
        return {}

    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = next((part for part in parts if "{" in part and "}" in part), cleaned)
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def extract_user_memory(token, user_message):
    extraction_prompt = (
        "Given this user message, extract any personal facts or preferences as a JSON object. "
        "Good keys include name, preferred_language, interests, communication_style, favorite_topics, "
        "location, or preferences. If none are present, return {} only.\n\n"
        f"User message: {user_message}\n\nJSON:"
    )
    raw_result = request_json_completion(token, extraction_prompt)
    return parse_json_object(raw_result)


def stream_assistant_reply(token, messages, memory):
    client = get_inference_client(token)
    stream = client.chat_completion(
        model=MODEL_ID,
        messages=build_model_messages(messages, memory),
        max_tokens=512,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        token_text = getattr(delta, "content", None)
        if token_text:
            time.sleep(0.02)
            yield token_text


def create_chat():
    created_at = datetime.now().isoformat()
    chat = {
        "id": f"chat_{datetime.now().timestamp()}",
        "title": "New Chat",
        "created_at": created_at,
        "messages": [],
    }
    save_chat(chat)
    return chat


def get_active_chat():
    active_chat_id = st.session_state.active_chat_id
    for chat in st.session_state.chats:
        if chat["id"] == active_chat_id:
            return chat
    return None


def update_chat_title(chat):
    for message in chat["messages"]:
        if message["role"] == "user":
            chat["title"] = message["content"][:30] or "New Chat"
            return


st.set_page_config(page_title="My AI Chat", layout="wide")

st.title("My AI Chat")
st.subheader("Task 3: User Memory")

token = load_hf_token()
if not token:
    st.error("Missing Hugging Face token. Add `HF_TOKEN` to your Streamlit secrets.")
    st.stop()

if "chats" not in st.session_state:
    st.session_state.chats = load_saved_chats()
    st.session_state.active_chat_id = (
        st.session_state.chats[0]["id"] if st.session_state.chats else None
    )

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

if st.session_state.active_chat_id and get_active_chat() is None:
    st.session_state.active_chat_id = (
        st.session_state.chats[0]["id"] if st.session_state.chats else None
    )

with st.sidebar:
    st.header("Chats")
    if st.button("New Chat", use_container_width=True):
        new_chat = create_chat()
        st.session_state.chats.insert(0, new_chat)
        st.session_state.active_chat_id = new_chat["id"]
        st.rerun()

    if st.session_state.chats:
        for chat in st.session_state.chats:
            row_col, delete_col = st.columns([5, 1])
            is_active = chat["id"] == st.session_state.active_chat_id
            label = f"{chat['title']}\n{format_timestamp(chat['created_at'])}"
            with row_col:
                if st.button(
                    label,
                    key=f"open_{chat['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_chat_id = chat["id"]
                    st.rerun()
            with delete_col:
                if st.button("✕", key=f"delete_{chat['id']}", use_container_width=True):
                    st.session_state.chats = [
                        existing_chat
                        for existing_chat in st.session_state.chats
                        if existing_chat["id"] != chat["id"]
                    ]
                    delete_chat_file(chat["id"])
                    if st.session_state.active_chat_id == chat["id"]:
                        st.session_state.active_chat_id = (
                            st.session_state.chats[0]["id"]
                            if st.session_state.chats
                            else None
                        )
                    st.rerun()
    else:
        st.info("No chats yet. Create one from the button above.")

    with st.expander("User Memory", expanded=True):
        if st.session_state.memory:
            st.json(st.session_state.memory)
        else:
            st.write("{}")

        if st.button("Clear Memory", use_container_width=True):
            st.session_state.memory = {}
            clear_memory()
            st.rerun()

active_chat = get_active_chat()
if active_chat is None:
    st.info("No active chat selected. Create a new chat from the sidebar.")
    st.stop()

if active_chat["messages"]:
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
else:
    st.info("This chat is empty. Send a message to begin.")

user_prompt = st.chat_input("Send a message")
if user_prompt:
    active_chat["messages"].append({"role": "user", "content": user_prompt})
    update_chat_title(active_chat)
    save_chat(active_chat)
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        try:
            assistant_message = st.write_stream(
                stream_assistant_reply(
                    token,
                    active_chat["messages"],
                    st.session_state.memory,
                )
            ).strip()
            if not assistant_message:
                assistant_message = "The model returned an empty streamed response."
                st.error(assistant_message)
        except Exception as error:
            assistant_message = interpret_hf_error(error)
            st.error(assistant_message)

    active_chat["messages"].append({"role": "assistant", "content": assistant_message})
    save_chat(active_chat)

    extracted_memory = extract_user_memory(token, user_prompt)
    if extracted_memory:
        st.session_state.memory = merge_memory(st.session_state.memory, extracted_memory)
        save_memory(st.session_state.memory)
