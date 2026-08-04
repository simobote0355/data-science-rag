from dotenv import load_dotenv
from retrieval import retrieve
import os
import streamlit as st
from groq import Groq

load_dotenv()

st.set_page_config(page_title="Data Science Assistant", page_icon="🤖")
st.title("🤖 Data Science Assistant")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask me about Data Science..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = retrieve(question, n_results=3)

    context = '\n\n'.join(
        f"Source: {r['metadata'].get('source', 'unknown')}\n{r['content']}"
        for r in results
    )

    message_system = {
        "role": "system",
        "content": (
            "You are an expert Data Science assistant. "
            "Answer based primarily on the following retrieved context. "
            "If the context does not contain enough information to answer, say so explicitly "
            "instead of making up an answer.\n\n"
            f"CONTEXT:\n{context}"
        )
    }

    message_groq = [message_system] + st.session_state.messages

    with st.chat_message("assistant"):
        answers = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=message_groq,
            temperature=0.3,
            stream=True
        )

        def get_text(stream):
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        
        text = st.write_stream(get_text(answers))

    with st.expander("📚 Sources consulted"):
        for r in results:
            fuente = r['metadata'].get('source', 'unknown')
            fragmento = r['content'][:200].strip()
            st.markdown(f"**{fuente}** (distancia: {r['distance']:.3f})")
            st.caption(f"{fragmento}...")
            st.divider()

    st.session_state.messages.append({"role": "assistant", "content": text})