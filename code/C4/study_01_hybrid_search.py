import json
import os
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 初始化设置
COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI = "http://localhost:19530"  # 服务器模式
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相对路径
BATCH_SIZE = 50

# 2. 连接 Milvus 并初始化嵌入模型
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
milvusClient = MilvusClient(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

# 3. 创建 Collection 存在删除
if milvusClient.has_collection(collection_name=COLLECTION_NAME):
    milvusClient.drop_collection(collection_name=COLLECTION_NAME)

# 4. 定义约束(表结构) 一个主键 七个varchar 一个稀疏 一个密集
fields = [
    FieldSchema(name = "id", dtype = DataType.INT64, is_primary = True, auto_id = True),
    FieldSchema(name = "img_id", dtype = DataType.VARCHAR, description = "图片id", max_length = 256), # VARCHAR限制 不指定直接报错
    FieldSchema(name = "path", dtype = DataType.VARCHAR, description = "图片路径", max_length = 256),
    FieldSchema(name = "title", dtype = DataType.VARCHAR, description = "标题" , max_length = 256),
    FieldSchema(name = "description", dtype = DataType.VARCHAR, description = "描述" , max_length = 256),
    FieldSchema(name = "category", dtype = DataType.VARCHAR, description = "分类", max_length = 256),
    FieldSchema(name = "location", dtype = DataType.VARCHAR, description = "位置", max_length = 256),
    FieldSchema(name = "environment", dtype = DataType.VARCHAR, description = "环境", max_length = 256),
    FieldSchema(name = "sparse_vector", dtype = DataType.SPARSE_FLOAT_VECTOR, description = "稀疏向量"),
    FieldSchema(name = "dense_vector", dtype = DataType.FLOAT_VECTOR   , description = "稠密向量" , dim=ef.dim["dense"]), # 稠密向量必须指定维度
]

# 如果集合不存在，则创建它及索引
if not milvusClient.has_collection(collection_name=COLLECTION_NAME):

    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields = fields, description = "关于龙的混合索引")
    milvusClient.create_collection(collection_name= COLLECTION_NAME, schema = schema)
    print("--> Collection 创建成功。")

    # 4. 创建索引
    # 创建稀疏索引 
    prepare_index_params = milvusClient.prepare_index_params()
    print("--> 正在为新集合创建索引...")

    print("创建稀疏向量索引ing")
    prepare_index_params.add_index(
        field_name = "sparse_vector",
        index_type = "SPARSE_INVERTED_INDEX",
        metric_type = "IP"
    )
    print("创建密集向量索引ing")
    prepare_index_params.add_index(
        field_name = "dense_vector",
        index_type = "AUTOINDEX",
        metric_type = "IP"
    )
    milvusClient.create_index(collection_name= COLLECTION_NAME, index_params= prepare_index_params)
    print("密集向量索引创建成功。")

# 获取集合
# 5. 加载数据并插入到milvus中 使用基于milvusClient的方式

milvusClient.load_collection(collection_name= COLLECTION_NAME)
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

status = milvusClient.get_collection_stats(collection_name=COLLECTION_NAME)
is_empty = status.get("row_count", 0) == 0
if is_empty:

    # 加载数据
    if not os.path.exists(path= DATA_PATH):
        raise FileNotFoundError(f"文件路径不存在:{DATA_PATH}")
    # with类似java中的tryResource 使用完资源后 会自动close 无需手动close
    # r 代表当前路径以 只读取reading方式打开 
    with open(DATA_PATH, "r", encoding= "utf-8") as f:
        dataset = json.load(f)

    docs, metadata = [], []
    for item in dataset:
        # parts使用列表定义 是为了更好的进行下面的join filter过滤
        parts = [
            item.get("title", ""),
            item.get("description", ""),
            item.get("location", ""),
            item.get("environment", ""),
        ]
        docs.append(" ".join(filter(None, parts)))
        metadata.append(item)
    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    print("--> 正在生成向量嵌入...")
    embeddings = ef(docs) 
    # 此处很关键 因为docs是列表 所以embeddings的["dense"][0] 就对应docs[0]的向量 embeddings的["dense"][1] 就对应docs[1]的向量
    # 获取稀疏 稠密向量
    sparse_vectors = embeddings["sparse"]
    dense_vectors = embeddings["dense"]
    # 此处不能直接使用embeddings["sparse"] 而是需要tocsr
    sparse_vectors = sparse_vectors.tocsr()
    print("--> 向量生成完成。")
    print("--> 正在分批插入数据...")
    
    print(f"sparse_vectors的类型是:", type(sparse_vectors))
    print(f"sparse_vectors的第一个元素是:",type(sparse_vectors[0]))
    print(f"sparse_vectors的第一个元素的shape是:",sparse_vectors[0].shape)

    # 构建entitiy
    insert_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        # 稀疏向量特殊处理
        sparse_row = sparse_vectors[i].tocoo()
        sparse_dict = {int(c): float(v) for c, v in zip(sparse_row.col, sparse_row.data)}
        print(f"sparse_dict的输出结果是:", sparse_dict)
        entity = {
            # 标量字段 (从原始 json 读取)
            "img_id": item.get("img_id", ""),
            "path": item.get("path", ""),
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "category": item.get("category", ""),
            "location": item.get("location", ""),
            "environment": item.get("environment", ""),
            #"sparse_vector" : sparse_vectors[i],
            "sparse_vector" : sparse_dict,
            "dense_vector" : dense_vectors[i]
        }
        insert_data.append(entity)

    # 遍历结束 插入数据
    milvusClient.insert(collection_name = COLLECTION_NAME, data = insert_data)
    # 刷新加载到内存
    milvusClient.flush(collection_name = COLLECTION_NAME)
    

# 6. 执行搜索 search_query , search_filter, topk
search_query = "查询悬崖上的龙"
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
limit = 5

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

# 查询文本转向量 因为search_query只有一个 所以只取数组0
search_query_embeddings = ef([search_query])
sparse_search_query_embeddings = search_query_embeddings["sparse"][0]
dense_search_query_embeddings = search_query_embeddings["dense"][0]

# 打印向量信息
print(f"稀疏向量信息:",sparse_search_query_embeddings)
print(f"稠密向量信息:",dense_search_query_embeddings)

# 定义搜索参数
search_params = {"metric_type" : "IP", "params" : {}}
# 稀疏的搜索向量
query_sparse = search_query_embeddings["sparse"][0].reshape(1, -1).tocoo()
query_sparse_dict = {
    int(col): float(val)
    for col, val in zip(query_sparse.col, query_sparse.data)
}


# 执行单独的搜索(稀疏向量)
sparse_search_result = milvusClient.search(
    collection_name= COLLECTION_NAME,
    data = [query_sparse_dict],
    anns_field="sparse_vector",
    limit= limit,
    search_params=search_params,
    filter=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)
print("\n--- [单独] 稀疏向量搜索结果 ---")
print("\n--- [单独] 稀疏向量搜索结果返回是: ---", sparse_search_result)


# 先执行单独的搜索(密集向量)
dense_search_result = milvusClient.search(
    collection_name= COLLECTION_NAME,
    data = [dense_search_query_embeddings],
    anns_field="dense_vector",
    limit= limit,
    search_params=search_params,
    filter=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)
print("\n--- [单独] 密集向量搜索结果 ---")
print("\n--- [单独] 密集向量搜索结果返回是: ---", dense_search_result)




# 执行混合搜索 
# 创建 RRF 融合器
rerank = RRFRanker(k=60)

# 创建搜索请求 两个AnnSearchRequest
sparse_search_request = AnnSearchRequest(
    data = [query_sparse_dict],
    anns_field = "sparse_vector",
    limit=limit,
    param=search_params,
    expr=search_filter
)

dense_search_request = AnnSearchRequest(
    data = [dense_search_query_embeddings],
    anns_field = "dense_vector",
    limit=limit,
    param=search_params,
    expr=search_filter
)

reqs = [sparse_search_request, dense_search_request]
# 执行混合搜索 使用milvusClient
hybrid_search_result = milvusClient.hybrid_search(
    collection_name = COLLECTION_NAME,
    reqs = reqs,
    ranker= rerank,
    limit= 15
)

print("\n--- 混合搜索的结果 ---", hybrid_search_result)

# 7. 清理资源
milvusClient.release_collection(collection_name=COLLECTION_NAME)
milvusClient.drop_collection(collection_name=COLLECTION_NAME)
print("\n--- 清理资源完成 ---")






