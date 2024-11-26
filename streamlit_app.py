import os
import openai
import xml.etree.ElementTree as ET
from pinecone import Pinecone, Index, ServerlessSpec
import streamlit as st

pinecone_api_key = st.secrets["PINECONE_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

pinecone_index_name = 'cvs'
pc = Pinecone(api_key=pinecone_api_key)

def create_pinecone_index():
    if pinecone_index_name in pc.list_indexes():
        pc.delete_index(pinecone_index_name)
    pc.create_index(pinecone_index_name, dimension=1536, metric="euclidean", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

index = pc.Index(pinecone_index_name)

def generate_embeddings(cv_text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=cv_text) 
    embedding = response['data'][0]['embedding']
    return embedding

def parse_xml_cv(xml_file, max_size=8192):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    document_text_element = root.find('.//DocumentText')
    if document_text_element is not None:
        text_data = document_text_element.text
        text_data = text_data.encode('utf-8')[:max_size].decode('utf-8', errors='ignore')
    else:
        text_data = ''

    #with open(xml_file, 'r') as file:
    #  text_data = file.read()
    return text_data

def parse_xml_cvs_folder(folder_path):
    cv_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            xml_file = os.path.join(folder_path, filename)
            cv_text = parse_xml_cv(xml_file)
            cv_data.append((filename, cv_text))
    return cv_data


#def store_embeddings_in_pinecone(embeddings):
#    index.upsert(vectors=embeddings)

def store_embeddings_in_pinecone(embeddings):
    vectors = [{'id': item['id'], 'values': item['embedding'], 'metadata': {'text': item['text']}} for item in embeddings]
    index.upsert(vectors=vectors)

def process_cvs_and_store_embeddings(folder_path):
    cv_data = parse_xml_cvs_folder(folder_path)
    embeddings = []
    
    for filename, cv_text in cv_data:
        embedding = generate_embeddings(cv_text)
        embeddings.append({'id': filename, 'embedding': embedding, 'text': cv_text})
    
    store_embeddings_in_pinecone(embeddings)
    


def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o", # text-embedding-ada-002",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Why these CVs match search query {text}"}
        ],
        #prompt="Why these CVs match search query " + text,
        #max_tokens=20000  # Adjust based on how much text you want
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def retrieve_embeddings_from_pinecone(ids):
    embeddings = []
    for id in ids:
        response = index.fetch(ids=[id])
        if 'vectors' in response and id in response['vectors']:
            vector_data = response['vectors'][id]
            embedding = vector_data['values']
            text = vector_data['metadata']['text']
            embeddings.append((embedding, text))
    return embeddings

def semantic_search(vector, num_results=5):
    results = index.query(vector=vector, top_k=num_results, include_values=True)
    return results


def search_cvs(query_text, num_results=5,threshold=0.5):
    vector = generate_embeddings(query_text)
    results = semantic_search(vector, num_results)
    cv_ids = [match['id'] for match in results['matches'] if match['score'] < threshold]
    print(cv_ids)
    embeddings_and_texts = retrieve_embeddings_from_pinecone(cv_ids)
    text_results = [text for _, text in embeddings_and_texts]
    
    answer = summarize_text( query_text + ' ' + "\n\n".join(text_results))
    return answer
    #return text_results

cvs_folder_path = 'cvs'

# Perform semantic search
# query = "Managing Director with over 25 years experience"
# query = "python developer in london"

with st.form("my_form"):
  st.title('Recruiter Pro')
  query = st.text_input("Search query")
  submit = st.form_submit_button('Do Search')
  if submit:
    results = search_cvs(query)
    st.text_area("Found", results, height=500)
#print(results)
