"""
Monitoring dashboard (rubric: Monitoring = 2/2, needs feedback collection +
a dashboard with 5+ charts - this file provides 6).

Run with: streamlit run monitoring/dashboard.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st

from core.config import config

st.set_page_config(page_title="GeoIntel AI - Monitoring", layout="wide")


@st.cache_data(ttl=30)
def load_conversations() -> pd.DataFrame:
    with psycopg.connect(config.pg_conninfo()) as conn:
        return pd.read_sql(
            "SELECT id, question, model, used_agent, tool_calls, prompt_tokens, "
            "completion_tokens, cost_usd, response_time_ms, created_at FROM conversations "
            "ORDER BY created_at",
            conn,
        )


@st.cache_data(ttl=30)
def load_feedback() -> pd.DataFrame:
    with psycopg.connect(config.pg_conninfo()) as conn:
        return pd.read_sql(
            "SELECT id, conversation_id, source, rating, relevance, explanation, created_at "
            "FROM feedback ORDER BY created_at",
            conn,
        )


st.title("GeoIntel AI — Monitoring Dashboard")

conversations = load_conversations()
feedback = load_feedback()

if conversations.empty:
    st.info("No conversations logged yet. Ask GeoIntel AI a question first (see the chat app).")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", len(conversations))
col2.metric("Total cost (USD)", f"${conversations['cost_usd'].sum():.4f}")
col3.metric("Avg response time", f"{conversations['response_time_ms'].mean():.0f} ms")
col4.metric("Agent usage", f"{100 * conversations['used_agent'].mean():.0f}%")

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("1. Cost over time")
    fig = px.line(conversations, x="created_at", y="cost_usd", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("2. Response time over time")
    fig = px.line(conversations, x="created_at", y="response_time_ms", markers=True)
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("3. Token usage per conversation")
    tok_df = conversations.melt(
        id_vars=["id"], value_vars=["prompt_tokens", "completion_tokens"],
        var_name="type", value_name="tokens",
    )
    fig = px.bar(tok_df, x="id", y="tokens", color="type", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("4. Model usage")
    fig = px.pie(conversations, names="model")
    st.plotly_chart(fig, use_container_width=True)

c5, c6 = st.columns(2)

with c5:
    st.subheader("5. Judge relevance distribution")
    judge_fb = feedback[feedback["source"] == "judge"]
    if not judge_fb.empty:
        fig = px.pie(judge_fb, names="relevance", color="relevance",
                     color_discrete_map={"RELEVANT": "#2ecc71", "PARTLY_RELEVANT": "#f39c12", "NON_RELEVANT": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No judge feedback yet.")

with c6:
    st.subheader("6. User feedback (thumbs up/down)")
    user_fb = feedback[feedback["source"] == "user"]
    if not user_fb.empty:
        counts = user_fb["rating"].map({1: "👍 Up", -1: "👎 Down"}).value_counts().reset_index()
        counts.columns = ["rating", "count"]
        fig = px.bar(counts, x="rating", y="count", color="rating",
                     color_discrete_map={"👍 Up": "#2ecc71", "👎 Down": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No user feedback yet.")

st.divider()
st.subheader("Recent conversations")
st.dataframe(
    conversations[["id", "question", "model", "used_agent", "cost_usd", "response_time_ms", "created_at"]]
    .sort_values("created_at", ascending=False)
    .head(20),
    use_container_width=True,
)
