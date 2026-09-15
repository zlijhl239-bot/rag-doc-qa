"""rebuild_parents.py — 重建 parent_chunks.pkl（修复父块重复存储的bug）"""
import os, pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
)
all_parents = []
pdf_files = sorted(f for f in os.listdir("./data/pdf_reports") if f.lower().endswith(".pdf"))[:5]
for fname in pdf_files:
    pages = PyPDFLoader(os.path.join("./data/pdf_reports", fname)).load()
    parents = parent_splitter.split_documents(pages)
    all_parents.extend(parents)
with open("./data/parent_chunks.pkl", "wb") as f:
    pickle.dump(all_parents, f)
print(f"重建完成: {len(all_parents)} 个父chunk（应为1261）")
