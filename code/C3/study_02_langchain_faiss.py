from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]

 # 构建文档
docs = [Document(page_content = i) for i in texts]

# 构建bge embedding模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 构建向量库 存储到本地
vectorstore = FAISS.from_documents(documents=docs, embedding = embeddings)
local_faiss_path = "./faiss_index_store"
vectorstore.save_local(folder_path = local_faiss_path)

 # 加载向量库 很关键 向量数据必须加载到内存中才能被查询 区别于关系数据库在磁盘中
loaded_vectorstore = FAISS.load_local(
    folder_path = local_faiss_path,
    embeddings = embeddings,
    allow_dangerous_deserialization = True
 )

# 构建查询
question = "LangChain是什么"

search_result  = loaded_vectorstore.similarity_search(
    query = question,
    k = 2   
)

print("FAISS 搜索的结果 search_result:", search_result)
"""
search_result: 
[
Document(id='a4baaff0-07bb-4c12-b55a-ec56e1bc77cc', metadata={}, page_content='LangChain是一个用于开发由语言模型驱动的应用程序的框架。'), 
Document(id='e3f801ad-2456-45e4-a4e1-14de8977f453', metadata={}, page_content='FAISS是一个用于高效相似性搜索和密集向量聚类的库。')
]
"""
 # 向量数据库查询

 # 看执行结果
