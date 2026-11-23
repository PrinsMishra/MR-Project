"""
Ingests and indexes academic policy documents into the vector store.
Processes PDF / text passages, generates dense embeddings, and stores them in AcademicVectorStore.
"""

from pathlib import Path
from ingestion.chunker import PolicyChunker
from retrieval.embeddings import PolicyEmbedder
from retrieval.vector_store import AcademicVectorStore

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_policies"
PDF_PATH = DATA_DIR / "academic_policy_nitech.pdf"
TXT_PATH = DATA_DIR / "academic_policy_nitech.txt"


def index_academic_policies():
    print("=" * 60)
    print("Indexing Academic Policy Documents into Vector Store...")
    print("=" * 60)

    chunker = PolicyChunker()

    if PDF_PATH.exists():
        print(f"Reading PDF: {PDF_PATH.name}")
        passages = chunker.chunk_pdf(PDF_PATH)
    else:
        print(f"Reading fallback text: {TXT_PATH.name}")
        # Parse text into paragraphs
        with open(TXT_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()
        pages = [{"page_number": 1, "document": TXT_PATH.name, "text": raw_text}]
        passages = chunker._fallback_paragraph_chunking(pages)

    print(f"Extracted {len(passages)} passages from policy documents.")

    embedder = PolicyEmbedder()
    print("Generating dense vector embeddings...")
    texts = [p.text for p in passages]
    embeddings = embedder.embed_batch(texts)

    vector_store = AcademicVectorStore()
    vector_store.upsert_passages(passages, embeddings)
    print(f"Successfully stored {len(passages)} passages in vector store.")
    print("Vector indexing completed successfully!")
    return len(passages)


if __name__ == "__main__":
    index_academic_policies()
