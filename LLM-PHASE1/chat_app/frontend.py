import streamlit as st
import requests

st.set_page_config(page_title="Chat with GPT", page_icon="💬")
st.title("💬 Chat with GPT (OpenAI)")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    payload = {
        "message": user_input,
        "history": st.session_state["history"]
    }
    try:
        resp = requests.post("http://localhost:8000/chat", json=payload)
        data = resp.json()
        if "reply" in data:
            reply = data["reply"]
            st.session_state["history"].append({"role": "user", "content": user_input})
            st.session_state["history"].append({"role": "assistant", "content": reply})
        elif "error" in data:
            st.error(f"API Error: {data['error']}")
        else:
            st.error("Unknown error from backend.")
    except Exception as e:
        st.error(f"Error: {e}")

# Display chat history
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**GPT:** {msg['content']}") 