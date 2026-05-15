import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

# torch开源模型

model_name = "BAAI/bge-base-en-v1.5"
model_weight = "../../models/bge/Visualized_base_en_v1.5.pth"

model = Visualized_BGE(
    model_name_bge = model_name,
    model_weight= model_weight
)

# 切换到评估模式 也就是推理模式
model.eval()

text = "datawhale开源组织的logo"
pic1 = "../../data/C3/imgs/datawhale01.png"
pic2 = "../../data/C3/imgs/datawhale02.png"

# 不要计算梯度
# 训练时 模型会计算梯度
# 推理时 
with torch.no_grad(): # 不要计算梯度
    # 作用域下的所有操作都不需要计算梯度 节省显存
    text_emb = model.encode(text = text)

    # 第一组 
    pic1_emb = model.encode(image = pic1)
    pic1_and_text_emb = model.encode(image = pic1, text = text)

    # 第二组
    pic2_emb = model.encode(image = pic2)
    pic2_and_text_emb = model.encode(image = pic2, text = text)

# 计算相似度
# @是python中的矩阵乘法运算符

# 纯图像对比
sim1 = pic1_emb @ pic2_emb.T
# 图文结合 vs 纯图像
sim2 = pic1_and_text_emb @ pic1_emb.T
# 图文结合 vs 纯文本
sim3 = pic2_and_text_emb @ text_emb.T
# 图文结合1 vs 图文结合2
sim4 = pic1_and_text_emb @ pic2_and_text_emb.T

print("=== 相似度计算结果 ===")
print(f"纯图像对比:{sim1}")
print(f"图文结合 vs 纯图像:{sim2}")
print(f"图文结合 vs 纯文本:{sim3}")
print(f"图文结合1 vs 图文结合2:{sim4}")

print("\n=== 嵌入向量信息 ===")
# 多模态向量维度 
"""
.shape 是 Python/NumPy/PyTorch 查看张量（tensor）维度的属性
返回的是一个 tuple，表示每个维度大小
例如：
[1, 1024] → 1 行 1024 列
[5, 1024] → 5 个向量，每个向量 1024 维
"""
print(f"多模态向量维度:{pic1_and_text_emb.shape}") # 
print(f"图片向量维度:{pic1_emb.shape}")
print(f"多模态向量维度前10:{pic1_and_text_emb[0][:10]}")
print(f"图片向量维度前10:{pic1_emb[0][:10]}")

"""
=== 相似度计算结果 ===
纯图像对比:tensor([[0.8318]])
图文结合 vs 纯图像:tensor([[0.8291]])
图文结合 vs 纯文本:tensor([[0.7916]])
图文结合1 vs 图文结合2:tensor([[0.9058]])

=== 嵌入向量信息 ===
多模态向量维度:torch.Size([1, 768])
图片向量维度:torch.Size([1, 768])
多模态向量维度前10:tensor([ 0.0360, -0.0032, -0.0377,  0.0240,  0.0140,  0.0340,  0.0148,  0.0292,
         0.0060, -0.0145])
图片向量维度前10:tensor([ 0.0407, -0.0606, -0.0037,  0.0073,  0.0305,  0.0318,  0.0132,  0.0442,
        -0.0380, -0.0270])
"""

