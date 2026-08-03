import chromadb
from embeddings import embed_query

client = chromadb.PersistentClient(path="./my_chroma_db")

collection = client.get_collection(name="knowledge_base")

def retrieve(question, n_results=3):
    results =  collection.query(query_embeddings=[embed_query(question)], n_results=n_results)

    return [
        {'content': doc, 'metadata': meta, 'distance': dist} 
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
    ]