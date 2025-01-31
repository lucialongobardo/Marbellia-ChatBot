from knowledgebase import create_index, load_retriever
from bs4 import BeautifulSoup
import requests
import serpapi
import os
import re
from transformers import BartTokenizer
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def query_pinecone(query, top_k, index, retriever):
    # generate embeddings for the query
    xq = retriever.encode([query], convert_to_tensor=True).tolist()[0]
    # search pinecone index for context passage with the answer
    xc = index.query(vector=xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
    # extract passage_text from Pinecone search result and add the <P> tag
    context = " ".join([f"<P> {m['metadata']['passage_text']}" for m in context['matches']])
    # contcatinate the query and context passages
    query = f"Pregunta del usuario: {query} \n Contexto para responder a la pregunta del usuario: {context}"
    return query

def get_question_context(query, top_k):
    # Creo el index
    _, index = create_index()
    # Load retriever model
    retriever = load_retriever()
    # search pinecone index for context passage with the answer
    context = query_pinecone(query, top_k, index, retriever)
    # format query with context passages
    query = format_query(query, context)    
    return query

# Función que realiza la búsqueda en Google y extrae el contenido relevante de la primera URL no patrocinada
def google_search_result(query):
    # Make a Google search
    s = serpapi.search(q=query, engine="google", location="Madrid, Spain", hl="es", gl="es", api_key=SERPAPI_API_KEY)
    # Get the first non-ad URL
    url = s["organic_results"][0]["link"]

    # Extraer el contenido de la página
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extraer el texto relevante de la página
    page_content = soup.get_text()

    page_content = re.sub(r'\n+', ' ', page_content)
    page_content = re.sub(r'\s+', ' ', page_content)

    # Cargar el tokenizador para BART
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Tokenizar el contenido para contar los tokens
    tokens = tokenizer.encode(page_content, truncation=True, max_length=1000)

    # Decodificar los tokens de nuevo en texto truncado si es necesario
    truncated_content = tokenizer.decode(tokens, skip_special_tokens=True)

    # Resume el contenido de la página
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    # Set the API headers
    headers = {"Authorization":"Bearer "+HUGGINGFACEHUB_API_TOKEN}
    # Make a request to the API
    response = requests.post(API_URL, headers=headers, json={"inputs":truncated_content})
    # Get the summary text from the response
    return response.json()[0]['summary_text'] if len(response.json())>0 else "No se ha podido obtener un resumen de la página"
