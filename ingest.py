from pathlib import Path 
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE, OVERLAP =800, 150

model = SentenceTransformer("intfloat/multilingual-e5-small")
client=chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

##fonction de decoupage
def chunk (text,size=CHUNK_SIZE,overlap=OVERLAP):
    step=size - overlap
    parts=[text[i:i+size] for i in range (0,len(text),step)]
    return [p for p in parts if p.strip]


###parcourir le pdf et fabriquer des chunks

ids, docs, metas = [], [], []

for pdf in Path("data/pdfs").glob("*.pdf"):
    for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
        text = page.extract_text() or ""
        for j, c in enumerate(chunk(text)):
            ids.append(f"{pdf.stem}-p{page_no}-{j}") ##pdf.stem nous donne le nom du pdf son extention 
            docs.append(c)
            metas.append({"source": pdf.name, "page": page_no})

## Calculer les vecteurs et stocker (on essaye de treansformer le texte en liste de nombres pour les comparer)
embs = model.encode(["passage: " + d for d in docs],
                    normalize_embeddings=True, show_progress_bar=True)
col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs.tolist())
print(len(docs), "chunks indexés")
