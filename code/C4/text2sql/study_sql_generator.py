import os
from typing import List, Dict, Any
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage, SystemMessage
from llama_index.llms.openai_like import OpenAILike  # 导入对接 LM Studio 的模块


class SimpleSQLGenerator:
    """简化的SQL生成器"""
    
    # init方法 -> 初始化模型配置
    def __int__(self):
        self.llm = OpenAILike(
            api_base="http://localhost:1234/v1", 
            api_key="lm-studio",  # LM Studio 不需要鉴权，但这里必须随便填一个非空字符串占位
            model="qwen2.5-7b-instruct",  # 替换为你 LM Studio 正在运行的模型名字
            is_chat_model=True
            )
        
    # build_context 方法
        # 基于milvusClient中的检索内容
            # 拼接content
            # 数据库描述信息
            # 表信息
            # ddl信息        
    def build_context(self, search_result: list[Dict[str, Any]]) -> str:

        if not search_result:
            return ''
        
        table_info = []
        ddl_info = []
        sql_info = []

        # sql ddl db
        for info in search_result:
            type = info.get("type", "")
            search_content = info.get("content", "")
            if type == 'db':                
                table_info.append(search_content)
            elif type == 'ddl':
                ddl_info.append(search_content)
            elif type == 'sql':                    
                sql_info.append(search_content)

        # 拼接结果集
        context = ''
        if table_info:
            context = "====== 表字段描述信息如下 ====== \n"     
            context += "\n".join(table_info)+ "\n\n"  

        if ddl_info:
            context += "====== 建表语句如下 ====== \n" 
            context += "\n".join(ddl_info)+ "\n\n"      

        if sql_info:
            context += "====== 表对应的样例sql如下 ====== \n"
            context += "\n".join(sql_info)+ "\n\n"  

        return context

    # generate_sql 方法
            # 构建系统提示词
            # 构建prompt
    def generate_sql(self, search_result: list[Dict[str, Any]], user_query: str) -> str:
        
        # 构建上下文
        context = self.build_context(search_result)

        # 构建prompt
        prompt = f""" 你是一个SQL专家。请根据以下信息将用户问题转换为SQL查询语句

数据库信息: 
{context}

用户问题:
{user_query}

要求:
1. 只返回SQL语句,不要包含任何解释信息
2. 确保SQL语法正确
3. 只能使用上下文中提供的名表和字段名
4. 如果需要JOIN 根据表结构进行合理关联

SQL语句："""
        # 生产环境中经常要求模型返回{"sql":"..."} 这种结构 省去了对结果集的各种正则提取

        result = self.llm.complete(prompt = prompt)
        print(f"模型的响应结果:", result.text)
        return result.text.strip()
        
    # fix_sql 方法
        # 构建系统提示词
        # 原始sql
        # 生成的ql
        # 报错内容    
    def fix_sql(self, source_sql : str, error_message : str, search_result :List[Dict[str, Any]]) -> str:

        # 构建上下文
        context = self.build_context(search_result)

        # 构建prompt
        prompt = f"""你是一个SQL专家 修复这个SQL
数据库信息:
{context}

原始SQL:
{source_sql}

错误信息:
{error_message}

1 返回修复后的SQL语句 不要包含任何解释信息
2 只能只有上下文中的表和字段信息 
3 确保语法正确
        """
        result = self.llm.complete(prompt=prompt)
        return result.text.strip()



    

    
    


    
   