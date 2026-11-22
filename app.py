import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from openai import OpenAI, APIError
from Bio import Entrez
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

st.title('Medical Literature Assistant')
question = st.text_input('Write question about miRNA:')
st.write('Your question:', question)

@st.cache_resource
def load_db():
    db_client = chromadb.PersistentClient(path = "./chroma_db")
    return db_client.get_or_create_collection(name = 'pubmed_stroke_rag')
pubmed_collection = load_db()

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranker = load_reranker()

load_dotenv()
my_key = os.getenv('GEMINI_API_KEY')

llm_client = OpenAI(
    api_key = my_key,
    base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
)
def fetch_and_index_pubmed(search_query, max_articles = 10):
    search_handle = Entrez.esearch(db = 'pubmed', term = search_query, retmax = max_articles, sort = 'relevance')
    search_results = Entrez.read(search_handle)
    search_handle.close()
    pmid_list = search_results['IdList']

    if not pmid_list:
        return 0

    fetch_handle = Entrez.efetch(db = 'pubmed', id = ','.join(pmid_list), rettype = 'medline', retmode = 'xml')
    records = Entrez.read(fetch_handle)
    fetch_handle.close()


    splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 100)
    ids, documents, metadatas = [], [], []
    added_articles = 0

    for article in records['PubmedArticle']:
        medline = article['MedlineCitation']
        pmid = f'PMID_{medline['PMID']}'
        art_data = medline['Article']
        title = str(art_data.get('ArticleTitle',''))

        if 'Abstract' in art_data and 'AbstractText' in art_data['Abstract']:
            abstract_parts = [str(part) for part in art_data['Abstract']['AbstractText']]
            full_abstract = ' '.join(abstract_parts)
            pub_date = art_data['Journal']['JournalIssue']['PubDate']
            year = int(pub_date.get('Year', 2024))

            chunks = splitter.split_text(full_abstract)
            for idx, chunk in enumerate(chunks):
                ids.append(f'{pmid}_chunk_{idx}')
                documents.append(chunk)
                metadatas.append({
                    'pmid': pmid,
                    'title': title,
                    'year': year,
                    'chunk': idx
                })
            added_articles += 1

    if ids:
        pubmed_collection.upsert(ids = ids, documents = documents, metadatas = metadatas)
    
    return added_articles
with st.sidebar:
    st.header("Settings")
    n_results = st.slider("Number of sources:", min_value = 1, max_value = 6, value = 3)
    min_year = st.slider("Published since year:", min_value = 2015, max_value = 2026, value = 2020)
    st.metric('Documents in base', pubmed_collection.count())
    with st.expander('View indexed articles'):
        all_data = pubmed_collection.get(include = ['metadatas'])
        unique_articles = {}
        for meta in all_data['metadatas']:
            pmid = meta['pmid']
            if pmid not in unique_articles:
                unique_articles[pmid] = f'**{pmid}** ({meta['year']}) - {meta['title'][:65]}...'
        
        if unique_articles:
            st.caption(f'Total unique articles : {len(unique_articles)}')
            for article_info in unique_articles.values():
                st.markdown(f'-{article_info}')
        else:
            st.info('Database is currently empty.')

    st.divider()
    st.subheader('Add From PubMed')
    pubmed_query = st.text_input('PubMed search phrase:', placeholder = 'e.g. miR-9-5p ischemic stroke')
    num_articles = st.number_input('Articles to fetch:', min_value = 1, max_value = 20, value = 5)

    if st.button('Fetch & Index') and pubmed_query:
        with st.spinner('Fetching from NCBI PubMed...'):
            count = fetch_and_index_pubmed(pubmed_query, max_articles= num_articles)
            if count > 0:
                st.success(f'Added {count} articles!')
                st.rerun()
            else:
                st.warning('No articles found for this phrase.')

if st.button('Search') and question:
    candidate_k = max(10, n_results * 3)
    results = pubmed_collection.query(
        query_texts = [question],
        n_results = candidate_k,
        where ={'year': {'$gte': min_year}}
    )
    if not results['documents'][0]:
        st.warning('No publications match the selected year. Use the slider to narrow down the year.')
        st.stop()

    docs = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[question, doc] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs, metas), key = lambda x: x[0], reverse = True)[:n_results]

    fragments = []
    for score, doc_text, meta in ranked:
        line = f"PMID: {meta['pmid']} | Year: {meta['year']} | Text: {doc_text}"
        fragments.append(line)

    context = '\n\n'.join(fragments)

    prompt = f"""
    Odpowiedz po polsku na pytanie użytkownika, opierając się WYŁĄCZNIE na poniższym kontekście.
    Przy każdej informacji podaj w nawiasie PMID oraz rok publikacji.
    Jeśli w kontekście nie ma odpowiedzi, napisz "Nie mam takich danych w bazie".

    Kontekst:
    {context}

    Pytanie: {question}
    """
    with st.spinner('Analyzing the publications...'):
        try:
            response = llm_client.chat.completions.create(
                model='models/gemini-3.5-flash-lite',
                messages=[{'role': 'user', 'content': prompt}]
            )
            st.write(response.choices[0].message.content)
        except APIError as e:
            st.error(f"API request failed (rate limit or connection issue): {e}")
            st.stop()

    st.subheader('Sources used from PubMed:')
    for score, doc_text, meta in ranked:
        with st.expander(f"{meta['pmid']} ({meta['year']}) - Relevance score: {score:.2f}"):
            st.write(doc_text)
