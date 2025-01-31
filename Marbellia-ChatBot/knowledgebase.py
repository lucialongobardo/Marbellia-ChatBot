#yt-dlp --write-subs --skip-download [youtube_url]
from pinecone import Pinecone
from pinecone import ServerlessSpec
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv, find_dotenv
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_ = load_dotenv(find_dotenv())
PINECONE_API_KEY  = os.getenv('PINECONE_API_KEY')

# Get youtube ids
def get_youtube_ids(route):
    yt_ids = []
    with open(route, 'r') as file:
        for line in file:
            yt_ids.append(line.split('=')[1].strip())
    return yt_ids

# Get transcriptions clean
def get_clean_transcriptions(yt_ids):
    trans_bruto = YouTubeTranscriptApi.get_transcripts(yt_ids, languages=['es','en'])
    return {k:" ".join([d['text'] for d in v if len(v)!=0]) for k, v in trans_bruto[0].items()}

# Create index
def create_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    spec = ServerlessSpec(cloud=cloud, region=region)
    index_name = "youtube-videos"
    if index_name not in pc.list_indexes().names():
        # create the index if it does not exist
        pc.create_index(index_name, dimension=768, metric="cosine", spec=spec)
    # connect to index we created
    index = pc.Index(index_name)
    return pc, index

# Load retriever model
def load_retriever():
    # set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load the retriever model from huggingface model hub
    retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base', device=device) #load the retriever model from HuggingFace. Use the flax-sentence-embeddings/all_datasets_v3_mpnet-base model
    return retriever

# Create embeddings and upsert them into the index
def create_embeddings(dicc, index, retriever):
    # Passage id
    p_id = 0
    # Itearte over transcriptions
    for yt_id, transcription in dicc.items():
        # Split the transcription into passages
        passages = [transcription[i:i+1000] for i in range(0, len(transcription), 1000)]
        # For each passage, create an embedding and upsert it into the index
        for passage in tqdm(passages):
            emb = retriever.encode(passage, convert_to_tensor=True)
            meta = {'yt_id': yt_id, 'passage_text': passage}
            to_upsert = [(str(p_id), emb.tolist(), meta)]
            _ = index.upsert(vectors=to_upsert)
            p_id += 1
            # upsert/insert these records to pinecone
            _ = index.upsert(vectors=to_upsert)
    # check that we have all vectors in index
    print(index.describe_index_stats())

"""
# Obtenemos las ids de los vídeos
ls_ids = get_youtube_ids('./urls.txt')

# Obtenemos las transcripciones de los vídeos
d_trans = get_clean_transcriptions(ls_ids)

# Creo el index
pc, index = create_index()

# Load retriever model
retriever = load_retriever()

# Poblate the database
create_embeddings(d_trans, index, retriever)
"""


