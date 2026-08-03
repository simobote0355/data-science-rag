import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_groq(message, context=None):
    messages = []
    if context:
        messages.append({'role': 'system', 'content': f'Eres un asistente que responde preguntas sobre Data Science basándote ÚNICAMENTE en el siguiente contexto. Si la respuesta no está en el contexto, di explícitamente que no tienes esa información, no uses tu conocimiento general. Si el contexto no responde exactamente la pregunta del usuario, no menciones temas relacionados,simplemente indica que no tienes esa información.\n\nContexto:\n{context}'})
    messages.append({'role': 'user', 'content': message})

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile', 
        messages=messages,
        temperature=0.4)

    return response.choices[0].message.content