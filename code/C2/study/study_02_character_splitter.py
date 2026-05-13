from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

"""
C2整体内容 文本分块
"""
file_path = "./data/C2/txt/蜂医.txt"

# 加载文档
loader = TextLoader(file_path=file_path , encoding="utf-8")
doc = loader.load()

# 文本分块
characterTextSplitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 100,
)

chunks = characterTextSplitter.split_documents(doc)

# 查看输出内容
print("总共分割的文档数量:", len(chunks))

for i, e in enumerate(chunks[:10]):
    print(f"当前的索引:{i},元素长度:{len(e.page_content)} 元素类型:{e.type}, 元素内容:{e.page_content}")