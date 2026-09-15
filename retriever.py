"""
retriever.py — 检索：混合召回 + RRF融合 + 父文档检索（Day3 v4）
query → 向量topK + BM25 topK(词干化) → 公司过滤(可选) → RRF融合排序 → 映射父chunk → 返回上下文
v4 改动：
  - BM25 词干化（PorterStemmer）：revenues→revenu, banking→bank, employees→employe
    （修复"Revenues vs revenue"、"banking vs bank"这类大小写/词形匹配不上的问题）
  - 必须与 rebuild_bm25_stem.py 建库时完全一致的 tokenize 逻辑
"""
import os, pickle, heapq, re
from nltk.stem import PorterStemmer
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
CHUNK_DIR = "./data"

_stemmer = PorterStemmer()

def _tokenize(text):
    """与 rebuild_bm25_stem.py 完全一致的词干化分词"""
    return [_stemmer.stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]

def load_index():
    emb = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(embedding_function=emb, persist_directory=CHROMA_DIR)
    with open(os.path.join(CHUNK_DIR, "parent_chunks.pkl"), "rb") as f:
        parent_chunks = pickle.load(f)
    with open(os.path.join(CHUNK_DIR, "small_meta.pkl"), "rb") as f:
        small_meta = pickle.load(f)
    with open(os.path.join(CHUNK_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    return vectorstore, parent_chunks, small_meta, bm25

def retrieve_parents(query, top_k=20, vectorstore=None, parent_chunks=None, small_meta=None, bm25=None, company=None):
    """混合召回 + 公司过滤(可选) + RRF融合 → 返回 (top索引, 父chunk列表, 完整融合排序)"""
    # ① 向量召回（已按距离升序，最相似在前）
    vec = vectorstore.similarity_search_with_score(query, k=top_k)
    vec_ranked = [r[0].metadata["parent_idx"] for r in vec]

    # ② BM25 召回（词干化：与建库时一致）
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    bm25_ranked = [small_meta["parent_idx"][i] for i in
                   heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])]

    # ③ 公司过滤（消融实验用，可选）
    if company:
        allowed = {i for i, p in enumerate(parent_chunks)
                   if str(p.metadata.get("source", "")).endswith(company + ".pdf")}
        vec_ranked = [pid for pid in vec_ranked if pid in allowed]
        bm25_ranked = [pid for pid in bm25_ranked if pid in allowed]

    # ④ RRF 融合
    K = 60
    fusion = {}
    for rank, pid in enumerate(vec_ranked):
        fusion[pid] = fusion.get(pid, 0) + 1.0 / (K + rank + 1)
    for rank, pid in enumerate(bm25_ranked):
        fusion[pid] = fusion.get(pid, 0) + 1.0 / (K + rank + 1)

    # ⑤ 完整融合排序 + 取前5
    ranked_all = [pid for pid, _ in sorted(fusion.items(), key=lambda x: -x[1])]
    top = ranked_all[:10]
    return top, [parent_chunks[i] for i in top], ranked_all

def search(query, top_k=20):
    """交互用：query → 返回 (索引, 上下文)"""
    vectorstore, parent_chunks, small_meta, bm25 = load_index()
    idxs, parents, _ = retrieve_parents(query, top_k=top_k, vectorstore=vectorstore,
                                        parent_chunks=parent_chunks, small_meta=small_meta, bm25=bm25)
    print(f"[RRF] 取前 {len(idxs)} 个父chunk: {idxs}")
    context = [p.page_content for p in parents]
    return idxs, context

if __name__ == "__main__":
    queries = [
        "What was the company's revenue in 2023?",
        "How many employees did the company have?",
        "What is the company's principal business?",
    ]
    for q in queries:
        idxs, ctx = search(q)
        print(f"\n问题: {q}")
        print(f"返回 {len(ctx)} 个父chunk，共 {sum(len(c) for c in ctx)} 字符")
        print(f"第一块开头: {ctx[0][:150] if ctx else '(空)'}\n")
