"""
Thin Streamlit chat client. Contains ZERO RAG logic - every question goes
through the FastAPI service over HTTP, exactly like a real frontend would.
This is what keeps the "add a nicer UI" stretch goal cheap: if this file
were doing its own retrieval, it would be a second implementation to keep in
sync with api/main.py instead of a thin client on top of it.

Run with: streamlit run ui/streamlit_app.py
(Requires the API to be running: uvicorn api.main:app --port 8000)
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="GeoIntel AI", page_icon="🛰️")
st.title("🛰️ GeoIntel AI")
st.caption("Ask questions about remote sensing & precision agriculture research")

if "messages" not in st.session_state:
    st.session_state.messages = []

use_agent = st.sidebar.toggle("Use agent (multi-step search)", value=True)
st.sidebar.markdown(f"API: `{API_URL}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "conversation_id" in msg:
            c1, c2 = st.columns([1, 10])
            if c1.button("👍", key=f"up_{msg['conversation_id']}"):
                requests.post(f"{API_URL}/feedback", json={"conversation_id": msg["conversation_id"], "rating": 1})
                st.toast("Thanks for the feedback!")
            if c2.button("👎", key=f"down_{msg['conversation_id']}"):
                requests.post(f"{API_URL}/feedback", json={"conversation_id": msg["conversation_id"], "rating": -1})
                st.toast("Thanks for the feedback!")

if question := st.chat_input("Ask about crop classification, UAV monitoring, land cover..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and thinking..."):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"question": question, "use_agent": use_agent}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.write(data["answer"])
                with st.expander(f"Sources ({len(data['retrieved_chunk_ids'])} chunks)"):
                    st.write(data["retrieved_chunk_ids"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "conversation_id": data["conversation_id"]}
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API at {API_URL}: {e}")
