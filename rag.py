import chromadb
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer("intfloat/multilingual-e5-small")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

##une fonction qui transforme notre qst en vecteur et chroma compare avec les vecteurs stockees et envoie les k plus proches
def retrieve(question, k=4):
    q = emb_model.encode(["query: " + question], normalize_embeddings=True)
    res = col.query(query_embeddings=q.tolist(), n_results=k)
    return [{"text": t, **m} for t, m in zip(res["documents"][0], res["metadatas"][0])]

if __name__ == "__main__":
    for r in retrieve("une question sur ton cours"):
        print(r["source"], "p.", r["page"], "\n", r["text"][:200], "\n---")


##la generation de la reponse
import ollama
LLM = "qwen2.5:3b"


def answer(question, k=4):
    chunks = retrieve(question, k)
    context = "\n\n".join(
        f"[{i+1}] ({c['source']}, p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks))
    prompt = f"""Réponds uniquement avec le contexte ci-dessous.
Si la réponse n'y est pas, dis : "Je ne sais pas d'après les documents."
Cite les sources avec [numéro].

Contexte :
{context}

Question : {question}"""
    r = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}])
    return {"answer": r["message"]["content"],
            "sources": [{"source": c["source"], "page": c["page"]} for c in chunks]}