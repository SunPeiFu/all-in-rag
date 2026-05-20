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
    def encode_query(self, text : str, image_path : str | None = None) ->list[float]:
        with torch.no_grad(): # no_grad 是啥意思
            if image_path and text:
                mm_embeddings = self.model.encode(image = image_path, text = text)
            elif image_path and not text:
                mm_embeddings = self.model.encode(image = image_path)
            elif text and not image_path:
                mm_embeddings = self.model.encode(text = text)
            else :
                return None
        return mm_embeddings.tolist()[0] # 此处的tolist区数组首个元素 是否合理
    
    def encode_image(self, image_path : str):
        with torch.no_grad(): # no_grad 是啥意思
            mm_embeddings = self.model.encode(image = image_path)
        return mm_embeddings.tolist()[0]     

# 3 把作为查询图和检索图 二者的结果做汇总 拼成一个大图
def visualize_results(retrieved_images: list, 
                      query_image_path: str | None = None, 
                      img_height: int = 300, 
                      img_width: int = 300, 
                      row_count: int = 3) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)


#######
# 1. 先处理和拼接检索到的图像（右侧主画布）
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        # 防止传入的图片数量超过了画布格子总数 (row_count * row_count)
        if row >= row_count:
            break
            
        start_row, start_col = row * img_height, col * img_width
        
        try:
            retrieved_pil = Image.open(img_path).convert("RGB")
            retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
            resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
            bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
            
            # 添加索引号
            cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            # 💡 增强健壮性：防止某张检索图片损坏导致整个程序崩溃
            cv2.putText(panoramic_image, f"Error: {i}", (start_col + 10, start_row + img_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 2. 💡 核心修改：判断是否存在查询图像
    if query_image_path is not None:
        # 如果有 Query 图片，创建左侧展示区，并用 hstack 水平拼接
        query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)
        
        try:
            query_pil = Image.open(query_image_path).convert("RGB")
            query_cv = np.array(query_pil)[:, :, ::-1]
            
            # 缩放并加上蓝色边框（注意：copyMakeBorder会增加像素，所以resize时先减去边框宽度）
            resized_query = cv2.resize(query_cv, (img_width - 20, img_height - 20))
            bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
            
            # 将 Query 居中放入左侧展示区的最下方一格
            start_r = img_height * (row_count - 1)
            query_display_area[start_r:start_r + img_height, :] = bordered_query
            cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception as e:
            cv2.putText(query_display_area, "Query Load Err", (10, panoramic_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 返回 [Query区 + 检索结果全景图]
        return np.hstack([query_display_area, panoramic_image])
    
    else:
        # 💡 如果没有传入 Query 图片，直接返回检索结果全景图
        return panoramic_image

#######

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

#v1版本查询 文字+图片
# query_text = "查询一个龙"
# query_image_path = image_list[0]
# query_vector = encoder.encode_query(text = query_text, image_path = query_image_path)

# v2版本查询 通过文字查询图像
query_text = "查询一只猴子"
query_vector = encoder.encode_query(text = query_text)
query_image_path = None

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
    visualize_result = visualize_results(retrieved_images = retrieved_images , query_image_path = query_image_path)
    # 定义输出结果
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, visualize_result)
    print(f"结果图像已保存到: {combined_image_path}")
    Image.open(combined_image_path).show() # 获取结果 调用拼图方法

    
# clear释放内存 删除集合
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")








