"""
data_loader.py — 年报RAG：数据加载与切片实验（Day1 v3 · 英文年报优化版）
借鉴 ERC2 冠军方案：
  1) 双粒度切片：小chunk(检索用) + 父chunk(生成用) → 为 Day2 父文档检索铺垫
  2) 切片策略对比：递归字符切分 vs 语义切分
  3) 割裂检测（英文规则）：找出被硬切断的chunk，作为父文档检索的动机

v3 改动（针对纯英文年报）：
  - separators 改为英文标点（. ! ? ; , ），中文的"。！？"对英文无效
  - 割裂检测改用英文规则：以 and/of/the/in 等连接词或逗号结尾 = 句子被切断
  - 新增：统计 SEC 封面/目录模板页数量（RAG价值低，Day3清洗时处理）
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

DATA_DIR = "./data/pdf_reports"

# SEC年报封面/目录常见开头（用于统计模板页，先不删除）
TEMPLATE_PREFIXES = ("table of contents", "united states securities", "securities and exchange")

def load_pdfs(folder):
    """加载文件夹下所有PDF，返回(正常文档, 抽不到文本的文件名列表)"""
    all_docs, empty_files = [], []
    if not os.path.exists(folder):
        print(f"[错误] 找不到目录 {folder}，请先完成 Step 1")
        return [], []
    pdf_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
    print(f"发现 {len(pdf_files)} 个 PDF 年报")
    for fname in pdf_files:
        path = os.path.join(folder, fname)
        try:
            pages = PyPDFLoader(path).load()
            text = "".join(p.page_content for p in pages)
            if len(text.strip()) < 50:
                empty_files.append(fname)
                print(f"[警告] {fname} 几乎抽不到文本({len(text)}字)，疑似扫描件，需OCR")
            else:
                all_docs.extend(pages)
                print(f"  已加载 {fname}: {len(pages)} 页")
        except Exception as e:
            print(f"[错误] {fname} 加载失败: {e}")
    print(f"合计加载 {len(all_docs)} 页")
    return all_docs, empty_files

def main():
    docs, empty = load_pdfs(DATA_DIR)
    if not docs:
        return

    # --- 统计SEC模板页（封面/目录，RAG价值低） ---
    template_count = sum(
        1 for d in docs
        if d.page_content.strip().lower().startswith(TEMPLATE_PREFIXES)
    )
    print(f"疑似SEC封面/目录模板页: {template_count}/{len(docs)}（Day3清洗时处理）")

    # --- 小chunk：检索粒度（~400字符），英文 separators ---
    small = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=60,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    ).split_documents(docs)

    # --- 父chunk：生成粒度（~2000字符） ---
    parent = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    ).split_documents(docs)

    print(f"\n小chunk数量: {len(small)}（检索用）")
    print(f"父chunk数量: {len(parent)}（生成用）")

    # --- 语义切分对比（只对前30页做实验，避免CPU太慢） ---
    print("\n语义切分开始（调用本地embedding，可能要1-3分钟）...")
    try:
        emb = OllamaEmbeddings(model="nomic-embed-text")
        sem = SemanticChunker(emb, breakpoint_threshold_type="percentile").split_documents(docs[:30])
        print(f"语义chunk数量: {len(sem)}（前30页）")
    except Exception as e:
        print(f"[警告] 语义切分失败: {e}")

    # --- 割裂检测 v2：英文规则 + 打印真实案例 ---
    cut_tails = (" and", " or", " of", " the", " in", " on", " at", " for", " with", " by", " to", ",", ";", ":", "—")
    cuts = []
    for idx, c in enumerate(small):
        if c.page_content.rstrip().endswith(cut_tails):
            cuts.append((idx, c))
    print(f"\n以『不完整结尾』的小chunk: {len(cuts)}/{len(small)}")
    for idx, c in cuts[:3]:
        nxt = small[idx + 1].page_content[:150] if idx + 1 < len(small) else "(无)"
        print(f"\n[割裂案例] {os.path.basename(c.metadata.get('source',''))} 第{c.metadata.get('page','?')}页")
        print("  chunk结尾: ..." + c.page_content[-120:])
        print("  下一chunk开头: " + nxt[:120])

if __name__ == "__main__":
    main()
