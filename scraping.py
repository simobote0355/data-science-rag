import requests
from bs4 import BeautifulSoup

def get_soup(url):
    headers = {"User-Agent": "Mozilla/5.0 (educational RAG project)"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error al descargar {url}")
        raise e

    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

def soup_sklearn(soup):
    def is_leaf(section_tag):
        hijos_section = section_tag.find_all('section', recursive=False)
        return len(hijos_section) == 0
    
    content = []
    all_sections = soup.find_all('section')
    sections = [s for s in all_sections if (is_leaf(s))]

    for section in sections:
        header = section.find(['h2', 'h3', 'h4'])
        tag_remove = header.find_all(class_=['section-number', 'headerlink'])
        for tag in tag_remove:
            tag.decompose()
        title = header.get_text().strip()
        
        tables = section.find_all('table')
        for table in tables:
            table.decompose()

        paragraphs = section.find_all('p')
        all_p = [p.get_text() for p in paragraphs]
        text = '\n'.join(all_p)
        content.append({'title': title, 'content': text, 'source': 'sklearn_metrics'})
    
    return content

def soup_rules_of_ml(soup):
    body = soup.find('div', class_='devsite-article-body')
    panel = body.find('devsite-key-takeaways-panel')
    if panel:
        panel.decompose()
    headers = body.find_all(['h2', 'h3','h4'])

    rules = []
    for header in headers:
        siblings = header.find_next_siblings()
        for i, sibling in enumerate(siblings):
            if sibling.name in ['h2', 'h3','h4']:
                content = [p.get_text() for p in siblings[:i]]
                text = "\n".join(content)
                rules.append({'title': header.get_text().strip(), 'content': text, 'source': 'rules_of_ml'})
                break
        else:
            content = [p.get_text() for p in siblings]
            text = '\n'.join(content)
            rules.append({'title': header.get_text().strip(), 'content': text, 'source': 'rules_of_ml'})
    
    return rules

def soup_glossary_ml(soup):
    panel = soup.find('devsite-key-takeaways-panel')
    if panel:
        panel.decompose()
    headers = soup.find_all('h2', class_='hide-from-toc')

    glossary = []
    for header in headers:
        if header.parent.name == 'p':
            start = header.parent
        else:
            start = header
        siblings = start.find_next_siblings()
        paragraphs = []
        for i, sibling in enumerate(siblings):
            if sibling.name == 'h2' or sibling.find('h2'):
                break
            content = sibling.get_text(separator=' ', strip=True)
            if content:
                paragraphs.append(content) 
            
        text = '\n'.join(paragraphs)

        glossary.append({'title': header.get_text().strip(), 'content': text, 'source': 'glossary_ml'})

    return glossary

def soup_feature_engineer(soup):
    text = soup.find('div', class_='html-chunk')

    web_content = []

    headers = text.find_all(['h2', 'h3'])
    prev = headers[0].find_previous_siblings()
    intro = [prev[-1].get_text(), prev[0].get_text()]
    text_intro = '\n'.join(intro)
    web_content.append({'title': 'Introduction', 'content': text_intro, 'source': 'feature_engineering'})
    
    for header in headers:
        siblings = header.find_next_siblings()
        for i, sibling in enumerate(siblings):
            if sibling.name in ['h2', 'h3']:
                content = [p.get_text() for p in siblings[:i]]
                text = '\n'.join(content)
                web_content.append({'title': header.get_text().strip(), 'content': text, 'source': 'feature_engineering'})
                break
        else:
            content = [p.get_text() for p in siblings]
            text = '\n'.join(content)
            web_content.append({'title': header.get_text().strip(), 'content': text, 'source': 'feature_engineering'})
    
    return web_content