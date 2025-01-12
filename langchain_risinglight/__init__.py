from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
import risinglight
import json

class RisingLightVectorStore(VectorStore):
    def __init__(self, embedding):
        self._embedding_function = embedding
        self._db = risinglight.open("risinglight.db")
        # self._db = risinglight.open_in_memory()
        self._dimensions = 0

    @property
    def embeddings(self):
        return self._embedding_function

    def add_texts(self, texts, metadatas=None):
        ids = []
        self._dimensions = len(self._embedding_function.embed_query(""))
        # check if the table already exists by using pg_tables
        if len(self._db.query("SELECT * FROM pg_catalog.pg_tables WHERE table_name = 'documents'")) == 0:
            self._db.query(f"CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, text TEXT, embedding VECTOR({self._dimensions}), metadata TEXT)")
        query_item = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            metadata = json.dumps(metadata)
            text = text.replace("'", "''")
            metadata = metadata.replace("'", "''")
            embedding = self._embedding_function.embed_query(text)
            query_item.append(f"('{i}', '{text}', '{embedding}', '{metadata}')")
            ids.append(i)
        self._db.query(f"INSERT INTO documents (id, text, embedding, metadata) VALUES {', '.join(query_item)}")
        print("num of documents:", self._db.query("SELECT count(*) FROM documents")[0][0])
        return ids

    def similarity_search(self, query, k=None):
        query_embedding = self._embedding_function.embed_query(query)
        if k is None:
            k = 4
        items = self._db.query(f"SELECT text, metadata FROM documents ORDER BY '{query_embedding}'::VECTOR({self._dimensions}) <-> embedding LIMIT {k}")
        documents = []
        for item in items:
            documents.append(Document(page_content=item[0], metadata=json.loads(item[1])))
        return documents

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None):
        vs = RisingLightVectorStore(embedding)
        vs.add_texts(texts, metadatas=metadatas)
        return vs
