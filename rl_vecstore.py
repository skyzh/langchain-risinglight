import re

import numpy as np
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
import risinglight
import json

class RisingLightVectorStore(VectorStore):
    def __init__(self, embedding):
        self._embedding_function = embedding
        self._db = risinglight.open("risinglight.db")
        self._db.query("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, text TEXT, embedding TEXT, metadata TEXT)")

    @property
    def embeddings(self):
        return self._embedding_function

    def add_texts(self, texts, metadatas=None):
        ids = []
        for i, text in enumerate(texts):
            embedding = self._embedding_function.embed_query(text)
            metadata = metadatas[i] if metadatas else {}
            metadata = json.dumps(metadata)
            text = text.replace("'", "''")
            metadata = metadata.replace("'", "''")
            self._db.query(f"INSERT INTO documents (id, text, embedding, metadata) VALUES ('{i}', '{text}', '{embedding}', '{metadata}')")
            ids.append(i)
        return ids

    def similarity_search(self, query):
        # query_embedding = self._embedding_function.embed_query(query)
        # vectors = np.vstack([query_embedding, self._embeddings])
        
        # similarity_matrix = cosine_similarity(vectors)
        # cosine_similarity_query = similarity_matrix[0, 1:]
        # best_match_index = np.argmax(cosine_similarity_query)

        # return [Document(page_content=self._texts[best_match_index], metadata=self._metadatas[best_match_index])]
        return []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None):
        vs = RisingLightVectorStore(embedding)
        vs.add_texts(texts, metadatas=metadatas)
        return vs
