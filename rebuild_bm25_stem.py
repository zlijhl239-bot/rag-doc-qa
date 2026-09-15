"""rebuild_bm25_stem.py — 词干化重建BM25：修复大小写/词形不匹配"""
import os, pickle, re
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
def tokenize(text):
    return [stemmer.stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]

with open("data/small_meta.pkl", "rb") as f:
    small_meta = pickle.load(f)

bm25 = BM25Okapi([tokenize(doc) for doc in small_meta["texts"]])
with open("data/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)
print(f"✅ 重建完成: {len(small_meta['texts'])} 个chunk，词干化BM25已保存")
