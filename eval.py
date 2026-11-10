"""Évaluation du retrieval : hit rate@k et MRR sur eval_questions.json.

Compare toutes les méthodes de recherche (dense / bm25 / hybrid, avec ou sans reranker)
sur l'index actuel (celui créé par ingest.py).

Usage :
    python eval.py                    -> tableau de comparaison
    python eval.py --misses           -> affiche aussi les questions ratées
    python eval.py --save 800/150     -> ajoute les lignes dans results.csv
"""
import argparse
import csv
import json
from pathlib import Path

from rag import col, retrieve

KS = (1, 3, 5)
CONFIGS = [(mode, rr) for rr in (False, True) for mode in ("dense", "bm25", "hybrid")]


def evaluate(data, mode, use_rerank, show_misses=False):
    """Rang de la bonne page pour chaque question (None = absente du top 5)."""
    ranks = []
    for item in data:
        results = retrieve(item["question"], max(KS), mode, use_rerank)
        rank = next((i + 1 for i, r in enumerate(results)
                     if r["source"] == item["source"] and r["page"] == item["page"]), None)
        ranks.append(rank)
        if rank is None and show_misses:
            print(f"  RATÉ ({mode}, rerank={use_rerank}) : {item['question']}")
            print(f"     attendu : {item['source']} p.{item['page']}"
                  f" | trouvé : {[(r['source'], r['page']) for r in results]}")
    n = len(ranks)
    scores = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    scores["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="CHUNKING", help="ex. 800/150 : ajoute les résultats à results.csv")
    parser.add_argument("--misses", action="store_true", help="affiche les questions ratées")
    args = parser.parse_args()

    with open("eval_questions.json", encoding="utf-8") as f:
        data = json.load(f)

    print(f"{len(data)} questions, {col.count()} chunks indexés\n")
    print(f"{'recherche':<18}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>7}")
    rows = []
    for mode, rr in CONFIGS:
        s = evaluate(data, mode, rr, args.misses)
        name = mode + (" + rerank" if rr else "")
        print(f"{name:<18}{s['hit@1']:>7.0%}{s['hit@3']:>7.0%}{s['hit@5']:>7.0%}{s['mrr']:>7.2f}")
        rows.append([args.save, name, col.count(), f"{s['hit@1']:.0%}", f"{s['hit@3']:.0%}",
                     f"{s['hit@5']:.0%}", f"{s['mrr']:.2f}"])

    if args.save:
        out = Path("results.csv")
        is_new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["chunking", "retrieval", "chunks", "hit@1", "hit@3", "hit@5", "mrr"])
            w.writerows(rows)
        print(f"\nRésultats ajoutés à {out}")


if __name__ == "__main__":
    main()
