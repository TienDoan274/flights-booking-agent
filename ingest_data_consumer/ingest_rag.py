from dotenv import load_dotenv
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from constant import *

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

import os
from typing import List, Dict
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.core import Document
from tqdm import tqdm

def read_text_files(folder_path: str) -> Dict[str, str]:
    text_contents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text_contents[filename] = file.read()
    return text_contents

def chunk_texts(texts: Dict[str, str]) -> List[Dict]:
    parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=OpenAIEmbedding())
    chunks = []
    
    for filename, content in texts.items():
        doc = Document(text=content, id_=filename)
        nodes = parser.get_nodes_from_documents([doc])
        for i, node in enumerate(nodes):
            chunk_id = f"{filename}_{i}"
            chunks.append({
                "id": chunk_id,
                "text": node.text,
                "filename": filename
            })
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    return [embed_model.get_text_embedding(text) for text in texts]

def index_to_qdrant(chunks: List[Dict], embeddings: List[List[float]], collection_name: str):
    client = QdrantClient(QDRANT_HOST)
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
    except Exception:
        pass

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        client.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(
                id=idx,  # Using sequential positive integers as IDs
                payload={"text": chunk["text"], "filename": chunk["filename"], "chunk_id": chunk["id"]},
                vector=embedding
            )]
        )

def index_to_elasticsearch(chunks: List[Dict], index_name: str):
    es = Elasticsearch(ELASTICSEARCH_HOST)
    
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)

    for chunk in chunks:
        es.index(
            index=index_name,
            id=chunk["id"],
            document={
                "text": chunk["text"],
                "filename": chunk["filename"],
                "chunk_id": chunk["id"]
            }
        )

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, '..', 'crawlers_producer', 'crawled_data_vn_airline')
    print(folder_path)
    texts = read_text_files(folder_path)
    chunks = chunk_texts(texts)
    embeddings = get_embeddings([chunk["text"] for chunk in chunks])
    
    index_to_qdrant(chunks, embeddings, "regulations")
    index_to_elasticsearch(chunks, "regulations")
    print("Ingest successfully !")
if __name__ == "__main__":
    main()