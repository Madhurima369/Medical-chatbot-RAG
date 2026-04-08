import streamlit as st
import requests

# ---- CONFIG ----
API_URL = "http://127.0.0.1:8000/chat"  # FastAPI endpoint

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical Chatbot")
st.write("Ask any medical question based on the knowledge base.")

# ---- Session State for Chat History ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Display Chat History ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- User Input ----
user_input = st.chat_input("Type your medical question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call FastAPI backend
    try:
        response = requests.post(API_URL, json={"question": user_input})
        result = response.json()

        answer = result.get("answer", "⚠️ No response from API.")

    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})