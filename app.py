from dotenv import load_dotenv
from retrieval import retrieve
import os
import streamlit as st
from groq import Groq

load_dotenv()

st.set_page_config(page_title="Asistente de Data Science", page_icon="🤖")
st.title("🤖 Asistente de Data Science")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Pregúntame sobre Data Science..."):
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
            "Eres un asistente experto en Data Science. "
            "Responde basándote principalmente en el siguiente contexto recuperado. "
            "Si el contexto no contiene información suficiente para responder, dilo explícitamente "
            "en vez de inventar una respuesta.\n\n"
            f"CONTEXTO:\n{context}"
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

    with st.expander("📚 Fuentes consultadas"):
        for r in results:
            fuente = r['metadata'].get('source', 'desconocida')
            fragmento = r['content'][:200].strip()
            st.markdown(f"**{fuente}** (distancia: {r['distance']:.3f})")
            st.caption(f"{fragmento}...")
            st.divider()

    st.session_state.messages.append({"role": "assistant", "content": text})