import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 定义模型配置

model_name = "qwen3.6-35b-a3b-mlx"
markdown_path = "./data/C1/markdown/easy-rl-chapter1.md"


# 定义模型配置
Settings.llm = OpenAILike(
    model= model_name,
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    api_base="http://127.0.0.1:1234/v1",
    is_chat_model=True
)

# 定义向量化
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 读取文件
docs = SimpleDirectoryReader(input_files=[markdown_path]).load_data()

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

print("打印llamindex的prompt是什么:", query_engine.get_prompts)
print("打印llamindex的查询结果:",query_engine.query("文中举了哪些例子?"))

# 输出结果
"""
打印llamindex的prompt是什么: <bound method PromptMixin.get_prompts of <llama_index.core.query_engine.retriever_query_engine.RetrieverQueryEngine object at 0x16d923410>>
打印llamindex的查询结果: 

文中主要列举了以下例子：
1. **选择餐馆**：去熟悉的餐馆是利用，搜索并尝试新餐馆是探索。
2. **做广告**：采用已知最优策略是利用，更换新策略测试效果是探索。
3. **挖油**：在已知地点挖掘是利用，去新地点勘探是探索（可能一无所获也可能发现大油田）。
4. **玩游戏**：如《街头霸王》中固定使用某种招式是利用，尝试新招式或“大招”是探索。
5. **$K$-臂赌博机（多臂赌博机）**：作为单步强化学习任务的理论模型。

"""


