from scraping import get_soup, soup_sklearn, soup_rules_of_ml, soup_glossary_ml, soup_feature_engineer
from chunking import get_tokens, split_chunk
from embeddings import get_embeddings
import json
import hashlib

SOURCES = [
    {'url': 'https://scikit-learn.org/stable/modules/model_evaluation.html', 'parser': soup_sklearn},
    {'url': 'https://developers.google.com/machine-learning/guides/rules-of-ml', 'parser': soup_rules_of_ml},
    {'url': 'https://developers.google.com/machine-learning/glossary', 'parser': soup_glossary_ml},
    {'url': 'https://www.geeksforgeeks.org/machine-learning/what-is-feature-engineering/', 'parser': soup_feature_engineer},
]

def load_sources(sources=SOURCES):
    content = []
    for source in sources:
        try:
            soup = get_soup(source['url'])
            content.extend(source['parser'](soup))
        except Exception as e:
            print(f"ERROR] Falló {source['url']}: {e}")
    return content

def deduplicate(content):
    seen = set()
    result = []
    for item in content:
        if item['content'] not in seen:
            seen.add(item['content'])
            result.append(item)
    return result

def add_tokens(content):
    for item in content:
        item['tokens'] = get_tokens(item['content'])
    return content

def chunking(content, threshold=400):
    final_content = []
    for item in content:
        final_content.extend(split_chunk(item, threshold=threshold))
    return final_content

def add_ids(content):
    for item in content:
        hash_id = hashlib.sha256(item['content'].encode()).hexdigest()[:8]
        item['id'] = f"{item['source']}_{hash_id}"
    return content

def save_json(content, path='knowledge_base.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

def main():
    print('Iniciando web scraping...')
    content = load_sources()

    print('Eliminando duplicados...')
    content = deduplicate(content)

    print('Obteniendo tokens...')
    content = add_tokens(content)

    print('Disminuyendo tamaño de chunks grandes...')
    content = chunking(content, threshold=400)

    print('Calculando vector para cada chunk...')
    content = get_embeddings(content)

    print('Añadiendo ID a cada chunk...')
    content = add_ids(content)

    print('Escribiendo JSON...')
    save_json(content)

    print('Ingestión finalizada.')

if __name__ == '__main__':
    main()