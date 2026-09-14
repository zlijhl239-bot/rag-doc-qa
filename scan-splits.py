"""参数扫描：不同 chunk_size 下，割裂率怎么变？"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = []
for f in sorted(os.listdir("./data/pdf_reports")):
    if f.lower().endswith(".pdf"):
        docs += PyPDFLoader(os.path.join("./data/pdf_reports", f)).load()

cut_tails = (" and", " or", " of", " the", " in", " on", " at", " for", " with", " by", " to", ",", ";", ":", "—")

print(f"{'chunk_size':>10} {'块数':>8} {'割裂数':>6} {'割裂率':>8}")
for size in [200, 400, 600, 800, 1000]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=size // 5,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    chunks = sp.split_documents(docs)
    cuts = sum(1 for c in chunks if c.page_content.rstrip().endswith(cut_tails))
    print(f"{size:>10} {len(chunks):>8} {cuts:>6} {cuts/len(chunks)*100:>7.1f}%")
