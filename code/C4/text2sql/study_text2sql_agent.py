import sqlite3
import os
from typing import Dict, Any, List, Tuple
from study_sql_generator import SimpleSQLGenerator
from study_knowledge_base import SimpleKnowledgeBase


class SimpleText2SQLAgent:

    """Text2SQL代理"""

    # 初始化方法
        # 知识库
        # sql生成器
        # 数据库地址
        # 链接
        # 最大尝试次数
        # topN
        # 最大limit限制
    def __int__(self, milvus_uri : str = '', topN : int = 5):
        self.knowledge_base = SimpleKnowledgeBase(milvus_uri = milvus_uri)
        self.sql_generator = SimpleSQLGenerator()
        self.db_path = None
        self.connection = None
        self.max_retry = 3
        self.top_n = topN
        self.limit = 5

    # 连接DB方法 入参 db_path
    def connect_db(self, db_path : str) -> bool:
        try:
            self.db_path = db_path
            self.connection = sqlite3.connect(db_path)
            print("成功连接到数据库")
            return True
        except Exception as e:
            print("连接到数据库失败: ", str(e))
            return False
        
    # 加载知识库
    def load_data(self):
        """加载知识库"""
        self.knowledge_base.load_data()
    
    # 查询方法
        # 增加while判断
        # 正常响应直接返回
        # 非正常判断是否到达最大重试次数 没有修复
    def query(self, user_question : str) -> Any:

        if not user_question:
            return None
        
        # 判断链接
        if not self.connection:
            return {
                "success": False,
                "error": "数据库未连接",
                "sql": None,
                "results": None
            }
        
        
        search_result = self.knowledge_base.search(content = user_question, top_k= self.top_n)
        if not search_result:
            return {
                "success": False,
                "error": "未检索到知识库信息",
                "sql": None,
                "results": None
            }
        
        sql = self.sql_generator.generate_sql(search_result=search_result, user_query= user_question)
        print(f"生成的原始sql:{sql}")

        retry_count = 0
        while retry_count < self.max_retry:
            is_success, result = self.execute_sql(sql)
            if is_success:
                print("SQL执行成功!")
                return {
                    "success": True,
                    "error": None,
                    "sql": sql,
                    "results": result,
                    "retry_count": retry_count
                }
            else:
                # 执行失败
                sql = self.sql_generator.fix_sql(source_sql= sql, error_message=result, search_result = search_result)
                retry_count +=1

        return {
                    "success": False,
                    "error": None,
                    "sql": sql,
                    "results": f"超过最大尝试次数",
                    "retry_count": retry_count
                }        

    # 执行SQL方法
        # 判断相关条件 没有添加 防止查穿db    
    def execute_sql(self, sql : str) -> Tuple[bool,Any]:

        """执行SQL语句"""
        try:
            cursor = self.connection.cursor()

            # 添加limit限制
            if sql.strip().upper().startswith('SELECT') and "LIMIT" not in sql.upper():
                # 把字符串后面的;去掉
                sql = f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"

            # 执行sql
            cursor.execute(sql)
            
            if sql.strip().upper().startswith('SELECT'):

                # 查询语句
                # cursor.description是数据库的游标属性 包含了每一列的元数据(列名 类型 大小等) 
                # 其中每个desc[0]即是 每一列第一个元素的列名
                columns = [desc[0] for desc in cursor.description]

                # 获取所有列的数据 注意此处无列名
                rows = cursor.fetchall()

                # 定义结果集
                results = []
                # rows的格式: [(1, "张三", 18), (2, "李四", 20)]
                for row in rows:
                    result_row = {}
                    # 此处即查出来的每一行数据
                    for i, value in enumerate(row):
                        result_row[i] = value
                    results.append(result_row)    

                cursor.close()
                # 注意此处 python的方法可以一次返回多个数值
                return True, {
                    "columns": columns,
                    "rows": results,
                    "count": len(results)
                }
            else:
                # 非查询语句
                self.connection.commit()
                cursor.close()
                return True, "SQL执行成功"
        
        except Exception as e:
            return False, str(e)

        
    
   
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """执行Text2SQL查询"""
        if not self.connection:
            return {
                "success": False,
                "error": "数据库未连接",
                "sql": None,
                "results": None
            }
        
        print(f"\n=== 处理查询: {user_question} ===")
        
        # 1. 从知识库检索
        print("检索知识库...")
        knowledge_results = self.knowledge_base.search(user_question, self.top_k_retrieval)
        print(f"检索到 {len(knowledge_results)} 条相关信息")
        
        # 2. 生成SQL
        print("生成SQL...")
        sql = self.sql_generator.generate_sql(user_question, knowledge_results)
        print(f"生成的SQL: {sql}")
        
        # 3. 执行SQL（带重试）
        retry_count = 0
        while retry_count < self.max_retry_count:
            print(f"执行SQL (尝试 {retry_count + 1}/{self.max_retry_count})...")
            
            success, result = self._execute_sql(sql)
            
            if success:
                print("SQL执行成功!")
                return {
                    "success": True,
                    "error": None,
                    "sql": sql,
                    "results": result,
                    "retry_count": retry_count
                }
            else:
                print(f"SQL执行失败: {result}")
                
                if retry_count < self.max_retry_count - 1:
                    print("尝试修复SQL...")
                    sql = self.sql_generator.fix_sql(sql, result, knowledge_results)
                    print(f"修复后的SQL: {sql}")
                
                retry_count += 1
        
        return {
            "success": False,
            "error": f"超过最大重试次数 ({self.max_retry_count})",
            "sql": sql,
            "results": None,
            "retry_count": retry_count
        }
    
    def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
        """执行SQL语句"""
        try:
            cursor = self.connection.cursor()
            
            # 添加LIMIT限制
            if sql.strip().upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
                sql = f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"
            
            cursor.execute(sql)
            
            if sql.strip().upper().startswith('SELECT'):
                # 查询语句
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result_row = {}
                    for i, value in enumerate(row):
                        result_row[columns[i]] = value
                    results.append(result_row)
                
                cursor.close()
                return True, {
                    "columns": columns,
                    "rows": results,
                    "count": len(results)
                }
            else:
                # 非查询语句
                self.connection.commit()
                cursor.close()
                return True, "SQL执行成功"
        
        except Exception as e:
            return False, str(e)
    
    def add_example(self, question: str, sql: str):
        """添加新的Q->SQL示例"""
        # 简化版本：直接保存到文件
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        qsql_path = os.path.join(data_dir, "qsql_examples.json")
        
        try:
            import json
            
            # 读取现有数据
            if os.path.exists(qsql_path):
                with open(qsql_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
            
            # 添加新示例
            data.append({
                "question": question,
                "sql": sql,
                "database": "sqlite"
            })
            
            # 保存
            with open(qsql_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"已添加新示例: {question}")
            
        except Exception as e:
            print(f"添加示例失败: {str(e)}")
    
    def get_table_info(self) -> List[Dict[str, Any]]:
        """获取数据库表信息"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_info = []
            for table in tables:
                table_name = table[0]
                
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                table_info.append({
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "nullable": not col[3],
                            "default": col[4],
                            "primary_key": bool(col[5])
                        }
                        for col in columns
                    ]
                })
            
            cursor.close()
            return table_info
            
        except Exception as e:
            print(f"获取表信息失败: {str(e)}")
            return []
    
    def cleanup(self):
        """清理资源"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("数据库连接已关闭")
        
        self.knowledge_base.cleanup()
        print("知识库已清理") 