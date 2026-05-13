from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 文档加载
file_path = "./data/C2/txt/蜂医.txt"
loader = TextLoader(file_path= file_path, encoding="utf-8")
docs = loader.load()

# 递归字符串切分 聪明之处在于如果切分后的内容还是太大 还会按照句子切 单词切等 业界主流
# 段落 句子 标题 单词 ...这个顺序
text_splitter = RecursiveCharacterTextSplitter(
    # 针对中英文混合文本，定义一个更全面的分隔符列表
    separators=["\n\n", "\n", "。", "，", " ", ""], # 按顺序尝试分割
    chunk_size=100,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(docs)

print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前10个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
