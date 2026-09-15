"""
retriever.py — 检索：混合召回 + 父文档检索（Day2）
query → 向量top10 + BM25 top10 → 合并去重 → 映射父chunk → 返回上下文
"""
import os, pickle, heapq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
CHUNK_DIR = "./data"

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

def search(query, top_k=10):
    vectorstore, parent_chunks, small_meta, bm25 = load_index()

    # ① 向量召回 → 命中小chunk的父chunk索引
    vec_results = vectorstore.similarity_search_with_score(query, k=top_k)
    vec_parents = [r[0].metadata["parent_idx"] for r in vec_results]
    print(f"[向量] 命中 {len(set(vec_parents))} 个父chunk")

    # ② BM25 召回
    scores = bm25.get_scores(query.split())
    bm25_top = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])
    bm25_parents = [small_meta["parent_idx"][i] for i in bm25_top]
    print(f"[BM25] 命中 {len(set(bm25_parents))} 个父chunk")

    # ③ 合并去重（混合召回）
    merged = sorted(set(vec_parents) | set(bm25_parents))
    print(f"[混合] 共 {len(merged)} 个父chunk 候选")

    # ④ ★父文档检索：返回父chunk内容
    context = [parent_chunks[i].page_content for i in merged[:5]]
    return merged[:5], context

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
