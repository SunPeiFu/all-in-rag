import os
from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from langchain_milvus import Milvus
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike  # 导入对接 LM Studio 的模块
from langchain_openai import ChatOpenAI
from pydantic import SecretStr



logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

# 使用bibiloader加载视频
    # 遍历加载好的文档
    # 重新赋值metadata
    # 构建组装bibi列表 元素是metadata
loader = BiliBiliLoader(video_urls = video_urls)
bili = []
try:
    docs = loader.load()
    for doc in docs:

        original = doc.metadata
        # 重新构建metadata去除无用信息
        metadata = {
            "title" : original.get("title", '未知标题'),
            "author" : original.get("author", '未知标题'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        # 重新赋值metadata
        doc.metadata = metadata
        bili.append(doc)
        

except Exception as e:
    print(f"加载视频文档异常 异常类型 {type(e)} 异常名: {type(e).__name__}异常内容:{e}")

if not bili:
    raise Exception("未加载到指定视频")


# Embeddings模型使用 -> model_name="BAAI/bge-small-zh-v1.5"
#embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# 创建向量存储 改成使用milvus的方式 存储到向量库中
vectorstore = Milvus.from_documents(
    documents = bili,
    embedding= embed_model,
    collection_name = "bili_video",
    connection_args = {"uri":"http://localhost:19530"},
    drop_old = True
)

# 配置元数据字段信息 标题 视频作者 视频观看次数 视频长度
# 作用是告诉llm 当前向量库collection中存储的数据表格式
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

llm = ChatOpenAI(
    model= "qwen2.5-7b-instruct",
    api_key=SecretStr("lm-studio"),
    base_url="http://127.0.0.1:1234/v1",
    model_kwargs={
        "max_tokens": 4096
    }
)


# 构建SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm = llm,
    vectorstore= vectorstore,
    metadata_field_info= metadata_field_info,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据", # 给LLM的prompt 告诉大模型当前向量库collection中存储的是什么内容
    enable_limit=True, # 限制条数 eg当你输入 时间最短的视频 隐含条件就是过滤只取1个 如不开启 可能无法准确识别带有数量限制的请求
    verbose = True # 打印llm输出的翻译详细日志 方便调试

)

# 5. 执行查询示例
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

# 使用retriever进行检索 查看检索内容
for i,query in enumerate(queries):
    result = retriever.invoke(query)
    print(f"第{i}次循环 执行查询的结果是:{result}")


