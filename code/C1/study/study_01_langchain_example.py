from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os

# 1 加载文档 数据准备 (非结构化/半结构化/结构化)数据
markdown_path = "./data/C1/markdown/easy-rl-chapter1.md"
loader = UnstructuredMarkdownLoader(markdown_path)
doc = loader.load()

# 2 文本分块 注意此处size和overlap参数大小 TODO SPF 此处需要研究下参数区别
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index = True # 记录chunk的元数据在在原始文档中的起始字符
)
chunks = text_spliter.split_documents(doc)

# 3 设置向量化模型 TODO SPF 此处需要研究下参数区别
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4 向量库指定embedding模型 并加载chunk 
vectorsStore = InMemoryVectorStore(embeddings)
vectorsStore.add_documents(chunks)

# 5 构建提示词 原始用户输入 和 向量检索后的结果
prompt = ChatPromptTemplate.from_template(
"""
请根据下面提供的上下文信息来回答问题 确保你的回答完全基于这些上下文 如果上下文中没有答案 直接告知 '从上下文中无法获取到必要信息 无法回复'

上下文:
{context}

问题:
{question}

回答:
"""
)

# 6 初始化模型
model = ChatOpenAI(
    model="zai-org/glm-4.7-flash",
    temperature=0.7,
    api_key=SecretStr("empty"),
    base_url="http://127.0.0.1:1234/v1",
    model_kwargs={
        "max_tokens": 4096
    }
)

# 7 构建问题检索
question = "文中举了哪些例子"
# 向量检索
retrieved_doc = vectorsStore.similarity_search(query=question, k = 3)
doc_content = "\n\n".join(doc.page_content for doc in retrieved_doc)

# 调用模型生成
answer = model.invoke(prompt.format(question = question, context = doc_content))
print("模型返回的answer类型是:\n", type(answer)) # <class 'langchain_core.messages.ai.AIMessage'>
print("模型返回content是:\n", answer.content)
print("模型返回response_metadata是:\n", answer.response_metadata)
print("模型返回type是:\n", answer.type)
print("模型返回name是:\n", answer.name)
print("模型返回id是:\n", answer.id)

# 模型返回的结果
"""
模型返回的answer类型是:
 <class 'langchain_core.messages.ai.AIMessage'>
模型返回content是:
 根据提供的上下文，文中举了以下例子：

1.  **DeepMind 研发的走路的智能体**：这是一个可以向前走，并能学习到举手保持平衡、在曲折道路上行走，以及加入扰动后变得更鲁棒等功能的智能体。
2.  **机械臂抓取**：通过使用多个机械臂进行训练，学习到一个适用于不同物体形状的统一抓取算法。
3.  **象棋选手**：在棋局结束时，赢棋得到正奖励，输棋得到负奖励。
4.  **选择餐馆**：利用是指去最爱的餐馆，探索是指尝试新餐馆。
5.  **做广告**：利用是指采取最优的广告策略，探索是指尝试新的广告策略。
6.  **挖油**：利用是指在已知的地方挖油，探索是指在新的地方挖油。
7.  **玩游戏**（如《街头霸王》）：利用是指采取固定策略（如蹲在角落出脚），探索是指尝试新招式（如放出大招）。
模型返回response_metadata是:
 {'token_usage': {'completion_tokens': 1500, 'prompt_tokens': 810, 'total_tokens': 2310, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 1285, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': None}, 'model_name': 'zai-org/glm-4.7-flash', 'system_fingerprint': 'zai-org/glm-4.7-flash', 'id': 'chatcmpl-1qkqhy4n8x125l9tjwv37pi', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}
模型返回type是:
 ai
模型返回name是:
 None
模型返回id是:
 run--5c1b81b8-aab7-464e-be56-f11779142b16-0

"""






#print("模型返回的答案是:\n", answer)
# 返回
