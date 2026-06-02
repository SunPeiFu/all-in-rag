import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike  # 导入对接 LM Studio 的模块


load_dotenv() # 加载环境变量使用 寻找隐藏的.env文件 

# 配置模型
Settings.llm = OpenAILike(
    api_base="http://localhost:1234/v1", 
    api_key="lm-studio",  # LM Studio 不需要鉴权，但这里必须随便填一个非空字符串占位
    model="qwen3.6-27b-ud-mlx",  # 替换为你 LM Studio 正在运行的模型名字
    is_chat_model=True
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1.加载数据并为每个工作表创建查询引擎和摘要节点
excel_file_path = '../../data/C3/excel/movie.xlsx'

# 读取excel
xls = pd.ExcelFile(excel_file_path)

# 定义 每个sheet定义查询引擎 k sheet名称 v 查询引擎
df_query_engines = {}
# 定义IndexNode集合
all_nodes = []

# 遍历excel列表
for sheet_name in xls.sheet_names:
    
    # 读取每个excel
    sheet = pd.read_excel(xls, sheet_name = sheet_name)

    # 解析成pandasquery
    # 理解成调用大模型翻译user的问题 然后执行python代码到excel里捞数 然后最终把结果增强生成
        # verbose开启详细人日志
    query_engine = PandasQueryEngine(df= sheet, llm = Settings.llm, verbose=True)

    year = sheet_name.replace('年份_', '')
    summary = f"这个表格包含了年份是{year}的电影信息 , 可以回答关于这一年电影的具体问题"
    index_node = IndexNode(text = summary, index_id = sheet_name)
    all_nodes.append(index_node)

    df_query_engines[sheet_name] = query_engine
    # 构建indexNode (摘要和索引id)

# 创建顶层索引 只包含索引和摘要
vector_index = VectorStoreIndex(all_nodes)

# 创建递归检索器
# 创建顶层检索器 用于在摘要节点中检索
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 创建递归检索器
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector" : vector_retriever},
    query_engine_dict = df_query_engines,
    verbose=True,
)

# 创建查询引擎
retriever_query_engine = RetrieverQueryEngine.from_args(retriever = recursive_retriever)

# 执行查询
query = "1993年上映的电影 评分最低的是哪一步电影"
result = retriever_query_engine.query(query = query)
print(f"查询的结果是:{result}")

# xls = pd.ExcelFile(excel_file)

# df_query_engines = {}
# all_nodes = []

# for sheet_name in xls.sheet_names:
#     df = pd.read_excel(xls, sheet_name=sheet_name)
    
#     # 为当前工作表（DataFrame）创建一个 PandasQueryEngine
#     query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)
    
#     # 为当前工作表创建一个摘要节点（IndexNode）
#     year = sheet_name.replace('年份_', '')
#     summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
#     node = IndexNode(text=summary, index_id=sheet_name)
#     all_nodes.append(node)
    
#     # 存储工作表名称到其查询引擎的映射
#     df_query_engines[sheet_name] = query_engine

# # 2. 创建顶层索引（只包含摘要节点）
# vector_index = VectorStoreIndex(all_nodes)

# # 3. 创建递归检索器
# # 3.1 创建顶层检索器，用于在摘要节点中检索
# vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# # 3.2 创建递归检索器
# recursive_retriever = RecursiveRetriever(
#     "vector",
#     retriever_dict={"vector": vector_retriever},
#     query_engine_dict=df_query_engines,
#     verbose=True,
# )

# # 4. 创建查询引擎
# query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# # 5. 执行查询
# query = "1994年评分人数最少的电影是哪一部？"
# print(f"查询: {query}")
# response = query_engine.query(query)
# print(f"回答: {response}")
