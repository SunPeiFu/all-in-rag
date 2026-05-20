import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# 1 基础配置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"

# 2 定义编码器 用于把图片转换成向量
class Encoder:

    def __init__(self,
                model_name : str,
                model_path : str):
        self.model = Visualized_BGE(model_name_bge = model_name, model_weight = model_path)
        self.model.eval() # 切换到评估模式

    def encode_query(self, image_path: str, text : str) ->list[float]:
        with torch.no_grad(): # no_grad 是啥意思
            mm_embeddings = self.model.encode(image = image_path, text = text)
        return mm_embeddings.tolist()[0] # 此处的tolist区数组首个元素 是否合理
    
    def encode_image(self, image_path : str):
        with torch.no_grad(): # no_grad 是啥意思
            mm_embeddings = self.model.encode(image = image_path)
        return mm_embeddings.tolist()[0]     

# 3 把作为查询图和检索图 二者的结果做汇总 拼成一个大图
def visualize_results(query_image_path: str, 
                      retrieved_images: list, 
                      img_height: int = 300, 
                      img_width: int = 300, 
                      row_count: int = 3) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return np.hstack([query_display_area, panoramic_image])

# 3. 初始化Encoder和Miluvs客户端
print("--> 初始化Encoder和Mivlus客户端 ing")
encoder = Encoder(model_name = MODEL_NAME, model_path = MODEL_PATH)
milvus_client = MilvusClient(uri= MILVUS_URI)

# 4 创建Milvus Collection
print(f"--> 正在创建miluvs Colleciton")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"--> 删除已存在的miluvs Colleciton {COLLECTION_NAME}")

# 5 使用glob文件查找 使用首个作为向量
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
first_image_path = image_list[0]
# encode_image后 返回的是维度数组
dim_array = encoder.encode_image(image_path = first_image_path)
dim = len(dim_array)

# 定义集合约束
fields = [
    FieldSchema(name = "id", dtype = DataType.INT64, is_primary = True, auto_id = True), # is_primary 是否是主键
    FieldSchema(name = "vector", dtype = DataType.FLOAT_VECTOR, dim = dim), # 向量字段一定需要维度
    FieldSchema(name = "image_path", dtype = DataType.VARCHAR, max_length = 128)
]
schema = CollectionSchema(fields = fields, description= "多模态图片检索约束")
print(f"schema的信息:{schema}")

# 创建集合&约束
milvus_client.create_collection(collection_name = COLLECTION_NAME, schema = schema)
print(f"正在创建集合ing {COLLECTION_NAME}")
collection_desc = milvus_client.describe_collection(collection_name = COLLECTION_NAME)
print(f"集合描述信息 {collection_desc}")

# 把所有图片向量化 存入milvus中
print(f"正在准备插入数据ing")
data_to_insert = []
for image_path in image_list:
    image_vector = encoder.encode_image(image_path = image_path)
    # 定义对象
    image_obj = {"vector" : image_vector, "image_path" : image_path}
    data_to_insert.append(image_obj)

if data_to_insert:
    count = milvus_client.insert(collection_name = COLLECTION_NAME, data = data_to_insert)  
    print(f"多模态数据写入{count}条数据")

# 创建&添加索引(向量字段)
prepare_index = milvus_client.prepare_index_params()
prepare_index.add_index(
    field_name = "vector",
    index_type="HNSW", # 索引类型 分层导航小世界算法
    metric_type="COSINE", # 度量类型 使用余弦相似度 为0 完全一致  为90 完全不一致
    params={"M": 16, "efConstruction": 256} # 超参数 来控制索引的质量
)

milvus_client.create_index(collection_name = COLLECTION_NAME, index_params = prepare_index)
print(f"milvus中创建索引成功")
index_desc = milvus_client.describe_index(collection_name = COLLECTION_NAME, index_name = "vector")
print(f"milvus中集合{COLLECTION_NAME}的索引信息:{index_desc}")

# 把向量数据加到内存中
milvus_client.load_collection(collection_name = COLLECTION_NAME)

# 执行多模态检索(同时传入文本&图片)
query_text = "查询一条龙"
query_image_path = image_list[0]
query_vector = encoder.encode_query(text = query_text, image_path = query_image_path)

search_result_list = milvus_client.search(
    collection_name = COLLECTION_NAME,
    data = [query_vector],
    output_fields = ["image_path"],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}} # 
)

print(f"查询的结果是: {search_result_list}")
retrieved_images = []
if search_result_list:
    for index, hit in enumerate(search_result_list[0]):
        # 输出id 距离 路径
        print(f"Top{index+1}的元素id{hit["id"]}, 距离:{hit["distance"]}, 图片路径:{hit["entity"]["image_path"]}")
        retrieved_images.append(hit["entity"]["image_path"])

print(f"正在可视化结果并清理图像")
if not retrieved_images:
    print("没有检索到图像")   
else:
    # 把查询的图像 和 检索召回的图像合成一个
    visualize_result = visualize_results(query_image_path = query_image_path, retrieved_images = retrieved_images)
    # 定义输出结果
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, visualize_result)
    print(f"结果图像已保存到: {combined_image_path}")
    Image.open(combined_image_path).show()
    


milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")

# 获取结果 调用拼图方法

# clear释放内存 删除集合












# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
#     FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
# ]

# # 创建集合 Schema
# schema = CollectionSchema(fields, description="多模态图文检索")
# print("Schema 结构:")
# print(schema)

# # 创建集合
# milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
# print(f"成功创建 Collection: '{COLLECTION_NAME}'")
# print("Collection 结构:")
# print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# # 5. 准备并插入数据
# print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
# data_to_insert = []
# for image_path in tqdm(image_list, desc="生成图像嵌入"):
#     vector = encoder.encode_image(image_path)
#     data_to_insert.append({"vector": vector, "image_path": image_path})

# if data_to_insert:
#     result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
#     print(f"成功插入 {result['insert_count']} 条数据。")

# # 6. 创建索引
# print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
# index_params = milvus_client.prepare_index_params()
# index_params.add_index(
#     field_name="vector",
#     index_type="HNSW",
#     metric_type="COSINE",
#     params={"M": 16, "efConstruction": 256}
# )
# milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
# print("成功为向量字段创建 HNSW 索引。")
# print("索引详情:")
# print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
# milvus_client.load_collection(collection_name=COLLECTION_NAME)
# print("已加载 Collection 到内存中。")

# # 7. 执行多模态检索
# print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
# query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
# query_text = "一条龙"
# query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

# search_results = milvus_client.search(
#     collection_name=COLLECTION_NAME,
#     data=[query_vector],
#     output_fields=["image_path"],
#     limit=5,
#     search_params={"metric_type": "COSINE", "params": {"ef": 128}}
# )[0]

# retrieved_images = []
# print("检索结果:")
# for i, hit in enumerate(search_results):
#     print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
#     retrieved_images.append(hit['entity']['image_path'])

# # 8. 可视化与清理
# print(f"\n--> 正在可视化结果并清理资源")
# if not retrieved_images:
#     print("没有检索到任何图像。")
# else:
#     panoramic_image = visualize_results(query_image_path, retrieved_images)
#     combined_image_path = os.path.join(DATA_DIR, "search_result.png")
#     cv2.imwrite(combined_image_path, panoramic_image)
#     print(f"结果图像已保存到: {combined_image_path}")
#     Image.open(combined_image_path).show()

# milvus_client.release_collection(collection_name=COLLECTION_NAME)
# print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
# milvus_client.drop_collection(COLLECTION_NAME)
# print(f"已删除 Collection: '{COLLECTION_NAME}'")
