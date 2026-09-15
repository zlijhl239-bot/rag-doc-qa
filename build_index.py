"""
build_index.py v2 — 增量建库：逐份PDF处理，每份完成打印进度，失败可重跑
"""
import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

DATA_DIR = "./data/pdf_reports"
CHROMA_DIR = "./chroma_db"
CHUNK_DIR = "./data"
MAX_PDFS = 5        # 先2验证，再改5全量

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
)
small_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=60,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
)

emb = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(embedding_function=emb, persist_directory=CHROMA_DIR)

all_parents = []        # 全局父chunk列表（按顺序，检索时按下标取）
parent_of_small = []    # 每个小chunk → 父chunk全局索引
corpus = []             # 每个小chunk的文本（与上面顺序同步，BM25用）

pdf_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))[:MAX_PDFS]

for fi, fname in enumerate(pdf_files):
    print(f"\n[{fi+1}/{len(pdf_files)}] 处理 {fname} ...")
    pages = PyPDFLoader(os.path.join(DATA_DIR, fname)).load()
    parents = parent_splitter.split_documents(pages)
    print(f"  父chunk: {len(parents)}")

    start = len(all_parents)          # 这份PDF的父chunk起始全局索引
    for p_idx, p in enumerate(parents):
        subs = small_splitter.split_documents([p])
        for s in subs:
            s.metadata["parent_idx"] = start + p_idx   # ★ 全局父chunk索引
            s.metadata["source"] = fname
            all_parents.append(p)                      # 父chunk按全局顺序存一份
            parent_of_small.append(start + p_idx)
            corpus.append(s.page_content)              # 与上面同步，保证顺序一致
        if subs:
            ids = [f"{fi}_{p_idx}_{i}" for i in range(len(subs))]
            vectorstore.add_documents(subs, ids=ids)   # ★ 增量入库
    print(f"  完成，当前Chroma集合大小: {vectorstore._collection.count()}")

# BM25（corpus 与 parent_of_small 同步构建，顺序严格一致）
bm25 = BM25Okapi([doc.split() for doc in corpus])

with open(os.path.join(CHUNK_DIR, "parent_chunks.pkl"), "wb") as f:
    pickle.dump(all_parents, f)
with open(os.path.join(CHUNK_DIR, "small_meta.pkl"), "wb") as f:
    pickle.dump({"parent_idx": parent_of_small, "texts": corpus}, f)
with open(os.path.join(CHUNK_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump(bm25, f)
print(f"\n全部完成 ✅ 小chunk总数: {len(parent_of_small)}")
