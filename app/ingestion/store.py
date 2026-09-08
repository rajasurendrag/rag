import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.config import (
  CHROMA_COLLECTION_NAME,
  CHROMA_PERSISTENCE_DIR,
  DOCUMENTS_DIR,
  EMBEDDING_MODEL,
)

from app.ingestion.document_processor import (
  load_documents,
  chunk_documents,
)


def create_embeddings():
  return OllamaEmbeddings(
    model=EMBEDDING_MODEL,
  )


def create_vector_store():
  embeddings = create_embeddings()

  return Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSISTENCE_DIR,
  )


def add_documents(documents):
  vector_store = create_vector_store()
  vector_store.add_documents(documents)

  return vector_store


def ingest_documents():
  documents = load_documents()
  chunks = chunk_documents(documents)

  return add_documents(chunks)


def compute_documents_hash():
  hasher = hashlib.sha256()

  for path in sorted(DOCUMENTS_DIR.glob("**/*.md")):
    hasher.update(path.relative_to(DOCUMENTS_DIR).as_posix().encode("utf-8"))
    hasher.update(path.read_bytes())

  return hasher.hexdigest()


def documents_hash_file():
  return CHROMA_PERSISTENCE_DIR / ".documents_hash"


def read_stored_documents_hash():
  hash_file = documents_hash_file()

  if not hash_file.exists():
    return None

  return hash_file.read_text().strip()


def write_documents_hash(digest):
  hash_file = documents_hash_file()
  hash_file.parent.mkdir(parents=True, exist_ok=True)
  hash_file.write_text(digest)


def initialize_vector_store():
  current_hash = compute_documents_hash()

  if current_hash == read_stored_documents_hash():
    return

  create_vector_store().reset_collection()
  ingest_documents()

  write_documents_hash(current_hash)


if __name__ == "__main__":
  ingest_documents()