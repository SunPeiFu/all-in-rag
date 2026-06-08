import json
import os
from typing import List, Dict, Any
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class SimpleKnowledgeBase:
    
    # init方法 
        # miluvs客户端
        # 初始化embedding模型
        # setup_collection -> 初始化miluvs中的集合
    def __init__(self, milvus_uri : str = "http://localhost:19530"):
        self.client = MilvusClient(milvus_uri)
        self.embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.collection_name = "knowledge_collection"
        self.setup_collection()

    # setup_collection方法
        # 存在删除 不存在创建
        # 设置集合中entity的约束
            # 字段 pk content type dense_vector
            # type 分别是 db描述说明 dll说明 sql范式说明
        # 创建索引
        # 创建集合
    def setup_collection(self):

        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name = "pk", description = "主键", dtype= DataType.INT32, is_primary = True, auto_id = True),
            FieldSchema(name = "content", description = "内容", dtype= DataType.VARCHAR, max_length = 256),
            FieldSchema(name = "type", description = "类型", dtype= DataType.VARCHAR, max_length = 256),
            FieldSchema(name = "dense_vector", description = "稠密向量" , dtype= DataType.FLOAT_VECTOR, dim = self.embedding_function.dim["dense"])
        ]    
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector" , 
            index_type = "IVF_FLAT",
            metric_type="IP" # 使用内积的方式计算
        )

        schema = CollectionSchema(fields = fields)

        self.client.create_collection(collection_name= self.collection_name, schema= schema, index_params=index_params)

    def load_data(self):
        # 当前python文件所在目录 -> 比如当前文件/Users/sunpeifualiyun.com/Desktop/agent_study_workspace/all-in-rag/code/C4/study_knowledge_base.py
        # 则current_file_path -> /Users/sunpeifualiyun.com/Desktop/agent_study_workspace/all-in-rag/code/C4
        current_file_path = os.path.dirname(__file__)

        # 生产级别的文件路径拼接方式, 即current_file_path/data
        data_dir = os.path.join(current_file_path, "data")

        # 三个文件路径
        db_path = os.path.join(data_dir, "db_descriptions.json")
        ddl_path = os.path.join(data_dir, "ddl_examples.json")
        qsql_path = os.path.join(data_dir, "qsql_examples.json")

    def load_db_content(self, path : str):

        with open(path, "r", encoding = "utf-8") as f:
            db_content = json.load(f)

        contents = []
        type = []    
        # 格式化 组装成字典  
        for element in db_content:

            table_name = element.get("table_name", "")
            table_description = element.get("table_description", "")
            columns = element.get("columns", "")
            
            content = f"表名:{table_name}\n"
            content += f"表描述信息:{table_description}\n"
            content += f"表字段信息:{columns}\n"

            contents.append(content)
            type.append("db")

        self.insert_data(contents, type)      
    

    def load_ddl_content(self, path : str):

        with open(path, "r", encoding = "utf-8") as f:
            ddl_content = json.load(f)

        contents = []
        type = []    
        # 格式化 组装成字典  
        for element in ddl_content:

            table_name = element.get("table_name", "")
            ddl_statement = element.get("ddl_statement", "")
            description = element.get("description", "")
            
            content = f"表名:{table_name}\n"
            ddl_statement += f"具体ddl信息:{ddl_statement}\n"
            description += f"ddl的描述:{description}\n"

            contents.append(content)
            type.append("ddl")  

        self.insert_data(contents, type)      

    def load_qsql_desc(self, path : str):

        with open(path, "r", encoding = "utf-8") as f:
            qsql_content = json.load(f)  

        contents = []
        type = []    

        # 格式化 组装成字典  
        for element in qsql_content:

            question = element.get("question", "")
            sql = element.get("sql", "")
            database = element.get("database", "")
            
            content = f":示例问题{question}\n"
            content += f"示例问题:{sql}\n"
            content += f"数据库名:{database}\n"

            contents.append(content)
            type.append("sql")  

        self.insert_data(contents, type)            

    def insert_data(self, contents: List[str], types: List[str]):

        if not contents:
            return
        
        # 向量化
        embeddings = self.embedding_function(contents)

        # 构建entity
        entities = []
        for i, element in enumerate(contents):
            content = element[i]
            type = types[i]
            dense_vector = embeddings["dense"][i]

            entity = {
                "content" : content,
                "type" : type,
                "dense_vector" :dense_vector
            }
            entities.append(entity)

        if not entities:
            return
        
        self.client.insert(collection_name=self.collection_name, data = entities)
        
    def search(self, content : str, top_k : int = 5) -> List[Dict[str, Any]]:

        query_embeddings = self.embedding_function([content])

    
        search_result = self.client.search(
            collection_name=self.collection_name,
            anns_field = "dense_vector",
            data = query_embeddings["dense"], # 此处必须是dense
            limit= top_k,
            search_params={"metric_type": "IP"},
            output_fields=["content", "type"]
        )

        # 此处为何是0 因为上面传入的self.embedding_function([content]) content只有一个 
        # search_result返回的是N组对应data查询的结果集 

        result_data = []
        for hit in search_result[0]:
            content = hit["entity"]["content"]
            type = hit["entity"]["type"]
            distance = hit["distance"]
            
            result_data.append({
                "content":content,
                "type":type,
                "distance":distance
            })
            
        return result_data

    
    def cleanup(self):
        """清理资源"""
        try:
            self.client.drop_collection(self.collection_name)
        except:
            pass 