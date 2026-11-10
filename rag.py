"""Recherche (retrieve) et génération de la réponse (answer).

Trois façons de chercher les passages :
- "dense"  : similarité des embeddings (sens de la phrase)
- "bm25"   : mots-clés (termes exacts, sigles, noms propres)
- "hybrid" : les deux, fusionnés avec Reciprocal Rank Fusion (RRF)
Option rerank=True : un cross-encoder relit les meilleurs candidats et les reclasse.

Génération : LLM local (Ollama) ou Claude (API Anthropic), choisi par la variable
d'environnement LLM_PROVIDER ("ollama" par défaut, ou "claude").
"""
import os
import re
import unicodedata

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMB_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingue, tourne sur CPU
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

emb_model = SentenceTransformer(EMB_MODEL)
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

# Tous les chunks en mémoire : nécessaire pour BM25 (quelques centaines de chunks, c'est léger)
_all = col.get(include=["documents", "metadatas"])
CHUNKS = {i: {"text": t, **m} for i, t, m in zip(_all["ids"], _all["documents"], _all["metadatas"])}
IDS = list(CHUNKS)


def tokenize(text):
    """Minuscules, sans accents, mots d'au moins 2 caractères."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [w for w in re.findall(r"\w+", text) if len(w) > 1]


bm25 = BM25Okapi([tokenize(CHUNKS[i]["text"]) for i in IDS])
_reranker = None  # chargé seulement si on s'en sert


def dense_search(question, n):
    q = emb_model.encode(["query: " + question], normalize_embeddings=True)
    return col.query(query_embeddings=q.tolist(), n_results=n)["ids"][0]


def bm25_search(question, n):
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(IDS)), key=lambda j: scores[j], reverse=True)[:n]
    return [IDS[j] for j in best]


def rrf(rankings, c=60):
    """Reciprocal Rank Fusion : chaque liste donne 1/(c + rang) à ses chunks, on additionne."""
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)


def rerank(question, ids):
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    scores = _reranker.predict([(question, CHUNKS[i]["text"]) for i in ids])
    return [i for _, i in sorted(zip(scores, ids), reverse=True)]


def retrieve(question, k=4, mode="dense", use_rerank=True, n_candidates=20):
    """Renvoie les k passages les plus pertinents : [{"text", "source", "page"}, ...]."""
    n = max(k, n_candidates)
    if mode == "dense":
        ids = dense_search(question, n)
    elif mode == "bm25":
        ids = bm25_search(question, n)
    elif mode == "hybrid":
        ids = rrf([dense_search(question, n), bm25_search(question, n)])[:n]
    else:
        raise ValueError(f"mode inconnu : {mode}")
    if use_rerank:
        ids = rerank(question, ids)
    return [CHUNKS[i] for i in ids[:k]]


INSTRUCTIONS = """Tu es un assistant qui répond à des questions sur des cours.
Réponds uniquement avec le contexte fourni, en français.
Si la réponse n'y est pas, dis : "Je ne sais pas d'après les documents."
Cite les sources avec [numéro]."""


class LLMError(RuntimeError):
    """Le LLM n'a pas pu répondre (service injoignable, clé invalide, refus...)."""


def generate_ollama(prompt):
    import ollama
    try:
        r = ollama.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": prompt}])
    except (ollama.ResponseError, ConnectionError) as e:
        raise LLMError(f"Ollama indisponible (est-il lancé ?) : {e}") from e
    return r["message"]["content"]


_claude = None


def generate_claude(prompt):
    import anthropic
    global _claude
    if _claude is None:
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise LLMError("Variable ANTHROPIC_API_KEY non définie : impossible d'appeler Claude")
        _claude = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY dans l'environnement
    try:
        r = _claude.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            system=INSTRUCTIONS,
            output_config={"effort": "low"},  # question-réponse courte : peu de réflexion suffit
            # si Claude refuse pour raison de sécurité, l'API relance sur un modèle de secours
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("Clé API Anthropic absente ou invalide (ANTHROPIC_API_KEY)") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Limite de requêtes Anthropic atteinte, réessaie plus tard") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Erreur de l'API Anthropic ({e.status_code}) : {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Impossible de joindre l'API Anthropic (réseau)") from e
    if r.stop_reason == "refusal":
        raise LLMError("Claude a refusé de répondre à cette question")
    return "".join(b.text for b in r.content if b.type == "text")


def answer(question, k=4, mode="dense", use_rerank=True, provider=None):
    chunks = retrieve(question, k, mode, use_rerank)
    context = "\n\n".join(
        f"[{i+1}] ({c['source']}, p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks))
    prompt = f"Contexte :\n{context}\n\nQuestion : {question}"
    provider = provider or LLM_PROVIDER
    if provider == "claude":
        text = generate_claude(prompt)
    elif provider == "ollama":
        text = generate_ollama(prompt)
    else:
        raise ValueError(f"LLM_PROVIDER inconnu : {provider}")
    return {"answer": text,
            "sources": [{"source": c["source"], "page": c["page"]} for c in chunks]}


if __name__ == "__main__":
    for r in retrieve("Qu'est-ce que la recherche opérationnelle ?"):
        print(r["source"], "p.", r["page"], "\n", r["text"][:200], "\n---")
