from unstructured.partition.auto import partition
from collections import Counter


# PDF文件路径
pdf_path = "./data/C2/pdf/rag.pdf"
content_type = "application/pdf"
# 加载文档
elements  = partition(
    filename = pdf_path,
    content_type= content_type
)

print("partition返回的type:", type(elements))

print("解析后元素的个数", len(elements))

print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

types = Counter(e.category for e in elements)
print(f"元素类型:{dict(types)}")


for i, e in enumerate(elements):
    print("当前元素index:{i}, 分类:{e.category}元素内容:{e}")
# 获取一共多少个字符

# 输出结果
"""
解析后元素的个数 279
解析完成: 279 个元素, 7500 字符
元素类型:{'Header': 22, 'Title': 195, 'UncategorizedText': 41, 'NarrativeText': 3, 'Footer': 15, 'ListItem': 3}
"""

