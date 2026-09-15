"""recall_at_k.py — 统计 gold 父块在 topK 内的命中率（K=1,5,10,20,50）"""
import os, json
from retriever import load_index, retrieve_parents

def main():
    vectorstore, parent_chunks, small_meta, bm25 = load_index()
    eval_set = json.load(open("data/eval_set.json", encoding="utf-8"))
    ks = [1, 5, 10, 20, 50]
    hits = {k: 0 for k in ks}
    tested = 0
    for q in eval_set:
        if not q.get("gold_pages"):
            continue
        _, _, ranked_all = retrieve_parents(
            q["question"], top_k=100, vectorstore=vectorstore,
            parent_chunks=parent_chunks, small_meta=small_meta,
            bm25=bm25, company=q["pdf"].replace(".pdf", ""))
        gold_idx = {i for i, p in enumerate(parent_chunks)
                    if str(p.metadata.get("source", "")).endswith(q["pdf"])
                    and p.metadata.get("page") in q["gold_pages"]}
        for k in ks:
            if set(ranked_all[:k]) & gold_idx:
                hits[k] += 1
        tested += 1
    for k in ks:
        print(f"Recall@{k:3d} = {hits[k]}/{tested} = {hits[k]/tested:.1%}")

if __name__ == "__main__":
    main()
