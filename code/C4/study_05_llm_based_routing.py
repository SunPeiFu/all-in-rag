import os
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableBranch
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1", # LM Studio 的本地地址
    model="google/gemma-4-26b-a4b-qat",  
    api_key=SecretStr("empty"),
    http_client=httpx.Client(trust_env=False), # 直连目标服务器 不走任何代理
       # 你的模型名称
    temperature=0
)

# StrOutputParser -> 解析并返回llm的回答
# 不同菜系的处理链

# 四川链
sichuan_prompt = ChatPromptTemplate.from_template(
    "你是一位川菜大厨。 请用正宗的川菜做法 回答关于[{question}]的问题"
)
sichuan_chain = sichuan_prompt | llm | StrOutputParser()

# 粤菜链
yuecai_prompt = ChatPromptTemplate.from_template(
    "你是一位粤菜大厨。 请用正宗的粤菜做法 回答关于[{question}]的问题"
)
yuecai_chain = yuecai_prompt | llm | StrOutputParser()

# 通用美食备用链
meishi_prompt = ChatPromptTemplate.from_template(
    "你是一位美食助手 请回答关于[{question}]的问题"
)
meishi_chain = meishi_prompt | llm | StrOutputParser()

# 路由
classifier_prompt = ChatPromptTemplate.from_template(
    """
    根据用户问题中提到的菜品 ,将菜系分类为:['川菜', '粤菜', '其他']。
    不要解释理由,只返回上述菜系分类中的一个结果
    问题:{question}
    """
)
classifier_chain = classifier_prompt | llm | StrOutputParser()

# 定义路由分支
router_branch = RunnableBranch(
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粤菜" in x["topic"], yuecai_chain),
    meishi_chain  # 默认选项
)

# 组合成完整路由链
# 1 
# {"topic" : classifier_chain , "question" : lambda x: x["question"]} 释义
# 多路并行 classifier_chain->返回一个单词 ,question原方不动透传 生成字典 -> {"topic":"川菜", "question":"麻婆豆腐怎么做?"}
full_router_chain = {"topic" : classifier_chain , "question" : lambda x: x["question"]} | router_branch
# 2 上一步的字典 传递到 -> router_branch 进行类似python中类似 if else的判断

# 3. 运行演示查询
demo_questions = [
    {"question": "麻婆豆腐怎么做？"},      # 应该路由到川菜
    {"question": "白切鸡的正宗做法是什么？"}, # 应该路由到粤菜
    {"question": "番茄炒蛋需要放糖吗？"}      # 应该路由到其他
]

for i, item in enumerate(demo_questions, 1): # 此处的1代表开始位置从几开始技术
    question = item["question"]
    print(f"\n--- 问题 {i}: {question} ---")
    
    try:
        # 获取路由决策
        topic = classifier_chain.invoke({"question": question})
        print(f"路由决策: {topic}")

        # 执行完整链
        result = full_router_chain.invoke(item)
        print(f"回答: {result}")
    except Exception as e:
        print(f"执行错误: {e}")
