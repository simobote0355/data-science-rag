import chromadb
import json

client = chromadb.PersistentClient(path="./my_chroma_db")

collection = client.get_or_create_collection(name="knowledge_base")

with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

ids = []
documents = []
embeddings = []
metadatas = []

for item in data:
    ids.append(item['id'])
    documents.append(item['content'])
    embeddings.append(item['embedding'])
    metadatas.append({'title': item['title'], 'source': item['source'], 'tokens': item['tokens']})

collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)