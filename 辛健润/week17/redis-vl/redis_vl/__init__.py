"""基于Redis的向量检索与智能缓存服务平台"""

__version__ = "0.1.0"

from redis_vl.SemanticCache import SemanticCache, EmbeddingCache
from redis_vl.SemanticMessageHistory import SemanticMessageHistory
from redis_vl.SemanticRouter import SemanticRouter
from redis_vl.schema import IndexSchema, IndexField, VectorIndex, FieldType, DistanceMetric, SchemaManager
from redis_vl.search import HybridSearch, SearchResult
from redis_vl.EmbeddingsCache import DashScopeEmbedding

__all__ = [
    "__version__",
    "SemanticCache",
    "EmbeddingCache",
    "SemanticMessageHistory",
    "SemanticRouter",
    "IndexSchema",
    "IndexField",
    "VectorIndex",
    "FieldType",
    "HybridSearch",
    "SearchResult",
]