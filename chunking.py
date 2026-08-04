from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")

def get_tokens(texto):
    tokens = tokenizer.encode(texto)
    return len(tokens)

def split_chunk(item, threshold=400):
    if item['tokens'] <= threshold:
        return [item]
    
    paragraphs = item['content'].split('\n')

    sub_chunks = []
    actual_paragraph = []
    actual_tokens = 0
    aux = 1

    for paragraph in paragraphs:
        paragraph_tokens = get_tokens(paragraph)

        if actual_tokens + paragraph_tokens > threshold and actual_paragraph:
            content = '\n'.join(actual_paragraph)
            sub_chunks.append({'title': f"{item['title']} (part {aux})", 'content': content, 'source': item['source'], 'tokens': actual_tokens})
            aux += 1
            actual_paragraph = []
            actual_tokens = 0

        actual_paragraph.append(paragraph)
        actual_tokens += paragraph_tokens
        
    if actual_paragraph:
        content = '\n'.join(actual_paragraph)
        sub_chunks.append({'title': f"{item['title']} (part {aux})", 'content': content, 'source': item['source'], 'tokens': actual_tokens})

    return sub_chunks