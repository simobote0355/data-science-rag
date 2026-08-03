from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-small')

def get_embeddings(chunks):
    text = ['passage: ' + chunk['content'] for chunk in chunks]
    embeddings = model.encode(text)

    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding.tolist()

    return chunks

def embed_query(question):
    text = 'query: ' + question
    embedding = model.encode(text)
    return embedding.tolist()