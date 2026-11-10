"""Indexation : PDF -> chunks -> vecteurs -> Chroma.

Usage :
    python ingest.py              -> chunks de 800 caractères, chevauchement 150
    python ingest.py 1500 280     -> autre taille / chevauchement

La collection est recréée à chaque lancement : aucun ancien chunk ne reste dans la base.
"""
import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 800
OVERLAP = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Découpe un texte en morceaux de `size` caractères qui se chevauchent de `overlap`."""
    step = size - overlap
    parts = [text[i:i + size] for i in range(0, len(text), step)]
    return [p for p in parts if p.strip()]  # enlève les chunks vides


def main():
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    client = chromadb.PersistentClient(path="chroma_db")

    # On repart d'une collection vide pour ne pas mélanger deux découpages
    if "docs" in [c.name for c in client.list_collections()]:
        client.delete_collection("docs")
    col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            text = page.extract_text() or ""
            for j, c in enumerate(chunk(text)):
                ids.append(f"{pdf.stem}-p{page_no}-{j}")
                docs.append(c)
                metas.append({"source": pdf.name, "page": page_no})

    # e5 attend le préfixe "passage: " pour les documents (et "query: " pour les questions)
    embs = model.encode(["passage: " + d for d in docs],
                        normalize_embeddings=True, show_progress_bar=True)
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs.tolist())
    print(f"{len(docs)} chunks indexés (taille {CHUNK_SIZE}, chevauchement {OVERLAP})")


if __name__ == "__main__":
    main()
