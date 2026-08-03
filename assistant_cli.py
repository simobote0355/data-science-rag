from client import ask_groq
from retrieval import retrieve

def get_context(chunks):
    parts = []
    for chunk in chunks:
        source = chunk['metadata']['source']
        parts.append(f"Source: {source}\n{chunk['content']}")
    
    return '\n\n'.join(parts)

def ask_assistant(question, n_results=3):
    chunks = retrieve(question, n_results)

    if not chunks:
        return 'No encontré información relevante en mi base de conocimiento para responder eso.'
    
    context = get_context(chunks)
    answer = ask_groq(question, context)

    return answer

if __name__ == '__main__':
    while True:
        question = input('Pregunta o salir: ')
        if question.lower() == 'salir':
            break
        answer = ask_assistant(question)
        print(f'\n{answer}\n')