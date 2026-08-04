import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_groq(message, context=None):
    messages = []
    if context:
        messages.append({'role': 'system', 'content': f'You are an assistant that answers Data Science questions based ONLY on the following context. If the answer is not in the context, explicitly say you don\'t have that information; do not use your general knowledge. If the context does not exactly answer the user\'s question, do not mention related topics, simply state that you don\'t have that information.\n\nContext:\n{context}'})
    messages.append({'role': 'user', 'content': message})

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile', 
        messages=messages,
        temperature=0.4)

    return response.choices[0].message.content