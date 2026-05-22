import os
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.openai_like import OpenAILike 

# 1. 配置模型
#Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))

Settings.llm = OpenAILike(
    api_base="http://localhost:1234/v1", 
    api_key="lm-studio",  # LM Studio 不需要鉴权，但这里必须随便填一个非空字符串占位
    model="qwen3.6-27b-ud-mlx",  # 替换为你 LM Studio 正在运行的模型名字
    is_chat_model=True
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
file_path = "../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"

# 2. 加载文档
docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

# 3. 创建节点与构建索引
# 3.1 句子窗口索引
sentenceWindowNodeParser = SentenceWindowNodeParser.from_defaults(
    window_metadata_key = "window",
    original_text_metadata_key = "original_text",
    window_size = 3
)
sentence_window_nodes = sentenceWindowNodeParser.get_nodes_from_documents(documents = docs)
sentence_windos_vector_store_index = VectorStoreIndex(nodes = sentence_window_nodes)

# 3.2 常规分块索引 (基准)
sentence_spliter = SentenceSplitter.from_defaults(
    chunk_size = 200,
    chunk_overlap = 50,
    include_metadata = True
)
sentence_nodes = sentence_spliter.get_nodes_from_documents(documents = docs)
sentence_vector_store_index = VectorStoreIndex(nodes = sentence_nodes)

# 4. 构建查询引擎
sentence_windos_query_engine = sentence_windos_vector_store_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
sentence_query_engine = sentence_vector_store_index.as_query_engine(similarity_top_k=2,)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"查询: {query}\n")

print("--- 句子窗口检索结果 ---")
sentence_windos_result = sentence_windos_query_engine.query(query = query)
print(f"sentence_windos_result查询结果:{sentence_windos_result}")
print("--- 常规检索结果 ---")
sentence_result = sentence_query_engine.query(query = query)
print(f"sentence_result查询结果:{sentence_result}")



