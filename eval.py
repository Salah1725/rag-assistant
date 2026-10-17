"""Évaluation du retrieval : hit rate@k et MRR sur eval_questions.json.

Usage :
    python eval.py              -> affiche les résultats
    python eval.py 800/150      -> affiche + ajoute une ligne dans results.csv
                                   (pour comparer les tailles de chunks)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rag import col, retrieve

KS = (1, 3, 5)
MAX_K = max(KS)

with open("eval_questions.json", encoding="utf-8") as f:
    data = json.load(f)

# Pour chaque question : rang de la bonne page dans les résultats (None = absente)
ranks = []
for item in data:
    results = retrieve(item["question"], MAX_K)
    rank = next(
        (i + 1 for i, r in enumerate(results)
         if r["source"] == item["source"] and r["page"] == item["page"]),
        None,
    )
    ranks.append(rank)
    if rank is None:
        print("RATÉ :", item["question"])
        print("   attendu :", item["source"], "p.", item["page"])
        print("   trouvé  :", [(r["source"], r["page"]) for r in results])

n = len(data)
n_chunks = col.count()

print(f"\n--- Résultats sur {n} questions ({n_chunks} chunks indexés) ---")
rates = {}
for k in KS:
    hits = sum(1 for r in ranks if r is not None and r <= k)
    rates[k] = hits / n
    print(f"hit rate@{k} = {hits}/{n} = {rates[k]:.0%}")

mrr = sum(1 / r for r in ranks if r is not None) / n
print(f"MRR = {mrr:.2f}")

# Détail par cours
by_file = defaultdict(list)
for item, r in zip(data, ranks):
    by_file[item["source"]].append(r)

print(f"\n--- Par cours (bonne page dans le top {MAX_K}) ---")
for name, rs in by_file.items():
    ok = sum(1 for r in rs if r is not None)
    print(f"{name} : {ok}/{len(rs)}")

# Optionnel : enregistrer une ligne pour le tableau du README
if len(sys.argv) > 1:
    label = sys.argv[1]
    out = Path("results.csv")
    is_new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["config", "chunks", "hit@1", "hit@3", "hit@5", "mrr"])
        w.writerow([label, n_chunks, f"{rates[1]:.0%}", f"{rates[3]:.0%}",
                    f"{rates[5]:.0%}", f"{mrr:.2f}"])
    print(f"\nLigne ajoutée à {out} (config : {label})")