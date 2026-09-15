"""
eval_retriever.py — 召回评测：Hit Rate（Day3 v3）
对每条评测问题跑混合检索（RRF融合），检查返回的父块是否包含标准答案所在页。
v3：retriever 返回三值；漏题打印 [诊断]（gold父块在完整融合排序中的名次）
"""
import json, os
from retriever import load_index, retrieve_parents

# 加在文件顶部（import 下面）
COMPANY_FULL = {
    "holley": "Holley Inc",
    "mercia": "Mercia Asset Management PLC",
    "tradition": "Compagnie Financiere Tradition SA",
    "tsx_y": "Yellow Pages Limited",
    "crossfirst": "CrossFirst Bankshares Inc",
}

def rewrite_query(question, company):
    """query增强：前置公司全名+报告语境，让检索更聚焦"""
    full = COMPANY_FULL.get(company, company)
    return f"{full} annual report 2022. {question}"


def main():
    vectorstore, parent_chunks, small_meta, bm25 = load_index()
    with open("data/eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    hits, tested, skipped = 0, 0, 0
    for q in eval_set:
        if not q.get("gold_pages"):
            skipped += 1
            continue

        idxs, parents, ranked_all = retrieve_parents(
            q["question"], top_k=20, vectorstore=vectorstore,
            parent_chunks=parent_chunks, small_meta=small_meta, bm25=bm25,
            company=q["pdf"].replace(".pdf", ""))   # 消融：公司过滤

        hit = any(
            str(p.metadata.get("source", "")).endswith(q["pdf"]) and p.metadata.get("page") in q["gold_pages"]
            for p in parents
        )
        tested += 1
        hits += hit
        hit_pages = [(os.path.basename(p.metadata.get("source", "?")), p.metadata.get("page")) for p in parents]
        print(f"{'✅' if hit else '❌'} {q['id']} [{q['company']}] gold页{q['gold_pages']} 命中{hit_pages}")
        if not hit:
            print(f"    问题: {q['question'][:80]}")
            # 诊断：gold页对应的父块在完整融合排序里的名次
            gold_pids = [i for i, p in enumerate(parent_chunks)
                         if str(p.metadata.get("source", "")).endswith(q["pdf"])
                         and p.metadata.get("page") in q["gold_pages"]]
            if not gold_pids:
                print(f"    [诊断] gold页{q['gold_pages']} 没有任何父块覆盖 → 切分层问题（页可能空/被跳过）")
            for gp in gold_pids:
                if gp in ranked_all:
                    print(f"    [诊断] gold父块#{gp} 在融合候选第 {ranked_all.index(gp)} 名（前5才返回）")
                else:
                    print(f"    [诊断] gold父块#{gp} 完全没进 vec/BM25 候选 → 召回层漏检")

    print(f"\n结果: {hits}/{tested} 命中 | Hit Rate = {hits/tested:.1%} | 跳过 {skipped} 条无gold")

if __name__ == "__main__":
    main()
