# Redis VL

基于 Redis 的向量检索与智能缓存服务平台。

## 功能特性

- **EmbeddingsCache**: 文本嵌入缓存，避免重复计算向量
- **SemanticCache**: 语义缓存，相似问题直接返回缓存结果
- **SemanticMessageHistory**: 对话历史管理，支持语义检索
- **SemanticRouter**: 意图路由，基于向量相似度自动分发
- **HybridSearch**: 混合检索，向量+关键词+元数据过滤

## 安装

```bash
pip install redis numpy dashscope
```

## 快速开始

### 配置环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key"
# Redis可选，默认 localhost:6379
```

### EmbeddingsCache - 嵌入缓存

```python
from redis_vl import EmbeddingCache

cache = EmbeddingCache()

# 缓存embedding
vector = cache.set("你好世界")

# 获取缓存的embedding（未命中则计算并缓存）
vector = cache.get_or_compute("你好世界")

# 批量操作
vectors = cache.mget_or_compute(["文本1", "文本2"])

# 查看缓存统计
print(cache.stats())  # {"size": 10, "ttl": 2592000}
```

### SemanticCache - 语义缓存

```python
from redis_vl import SemanticCache

cache = SemanticCache(threshold=0.85)

# 缓存问答对
cache.set("今天天气好吗", "天气很好，阳光明媚！")

# 获取缓存（相似问题会命中）
result, score = cache.get("今天天气怎么样")
# result: "天气很好，阳光明媚！" | score: 0.92
```

### SemanticMessageHistory - 对话历史

```python
from redis_vl import SemanticMessageHistory

history = SemanticMessageHistory(session_id="user123")

# 添加对话
history.add_user_message("你好")
history.add_assistant_message("你好，我是AI助手")
history.add_user_message("今天天气怎么样")
history.add_assistant_message("今天天气很好")

# 获取最近2轮对话
messages = history.get_messages(num_turns=2)
print(history.to_llm_format())

# 保存到Redis
history.save()

# 语义检索相关历史
related = history.search("天气")
```

### SemanticRouter - 意图路由

```python
from redis_vl import SemanticRouter

router = SemanticRouter()

# 注册路由（装饰器方式）
@router.route("天气查询")
def handle_weather(query):
    return "查询天气功能"

@router.route("新闻查询")
def handle_news(query):
    return "查询新闻功能"

# 路由分发
result, intent = router.dispatch("今天天气怎么样")
# result: "查询天气功能" | intent: "天气查询"

# 保存到Redis
router.save()
```

### HybridSearch - 混合检索

```python
from redis_vl import HybridSearch, IndexSchema, IndexField, VectorIndex, FieldType
import numpy as np

# 创建索引
schema = IndexSchema(
    index_name="my_index",
    fields=[
        IndexField(name="content", field_type=FieldType.TEXT),
        IndexField(name="category", field_type=FieldType.TAG),
        IndexField(
            name="vector",
            field_type=FieldType.VECTOR,
            vector_index=VectorIndex(
                index_type="HNSW",
                dimension=1024,
            )
        ),
    ]
)

search = HybridSearch(index_name="my_index")

# 添加文档（需要自行构建向量）
doc = {
    "content": "今天天气很好",
    "category": "weather",
    "vector": np.random.rand(1024).astype(np.float32)
}
search.add_document("doc1", doc["vector"], payload={"content": doc["content"], "category": doc["category"]})

# 检索
results = search.search("天气怎么样", top_k=5, filters={"category": "news"})
for r in results:
    print(r.id, r.score, r.payload)

# 纯文本检索
results = search.text_search("天气", top_k=5)
```

## API

### Config

```python
from redis_vl.config import get_config, set_config, Config, RedisConfig

# 自定义配置
config = Config(
    redis=RedisConfig(host="localhost", port=6379, password=""),
    dashscope=DashScopeConfig(api_key="sk-xxx", model="text-embedding-v3"),
    vector=VectorConfig(dimension=1024),
)
set_config(config)
```

## License

MIT