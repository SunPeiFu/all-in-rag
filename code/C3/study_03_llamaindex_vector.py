from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike  # 导入对接 LM Studio 的模块

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

Settings.llm = OpenAILike(
    api_base="http://localhost:1234/v1", 
    api_key="lm-studio",  # LM Studio 不需要鉴权，但这里必须随便填一个非空字符串占位
    model="qwen2.5-7b-instruct",  # 替换为你 LM Studio 正在运行的模型名字
    is_chat_model=True
)

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "新华字典里收录了所有的常见汉字"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

# 4 执行检索 查询
query_engine = index.as_query_engine()
response = query_engine.query("查询汉字需要使用什么")

print("LlamaIndex resposne的内容是:", response)
# 注意看: 此处已经返回的是llm 生成的内容 包含标准答案
"""
查询汉字需要使用新华字典。
"""
# 完整的内容在同级目录 llamaindex_index_store文件夹中
