"""语义路由 - SemanticRouter"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import redis

from redis_vl.client import get_redis_client, RedisClient
from redis_vl.EmbeddingsCache import DashScopeEmbedding


RouteHandler = Callable[[str], Any]
"""路由处理器类型"""


class SemanticRouter:
    """
    语义路由器 - 基于向量相似度的意图识别

    使用示例:
        router = SemanticRouter()

        # 注册路由
        @router.route("天气查询")
        def handle_weather(query):
            return "查询天气功能"

        @router.route("新闻查询")
        def handle_news(query):
            return "查询新闻功能"

        # 路由处理
        result = router.dispatch("今天天气怎么样")
    """

    def __init__(
        self,
        namespace: str = "sem_router",
        threshold: float = 0.7,
        default_handler: Optional[RouteHandler] = None,
        embedding_model: Optional[DashScopeEmbedding] = None,
        client: Optional[RedisClient] = None,
    ):
        self.namespace = namespace
        self.threshold = threshold
        self.default_handler = default_handler or (lambda x: "未知意图")
        self.embedding = embedding_model or DashScopeEmbedding()
        self.client = client or get_redis_client()

        self.routes: Dict[str, RouteHandler] = {}
        self.intent_vectors: Dict[str, np.ndarray] = {}

    def route(self, intent: str):
        """
        路由装饰器

        Args:
            intent: 意图名称
        """
        def decorator(func: RouteHandler) -> RouteHandler:
            self.routes[intent] = func
            return func
        return decorator

    def register(self, intent: str, handler: RouteHandler) -> "SemanticRouter":
        """
        注册路由

        Args:
            intent: 意图名称
            handler: 处理函数
        """
        self.routes[intent] = handler
        return self

    def _make_key(self, intent: str) -> str:
        """生成key"""
        return f"{self.namespace}:intent:{intent}"

    def _make_set_key(self) -> str:
        """生成集合key"""
        return f"{self.namespace}:routes"

    def save(self) -> bool:
        """保存路由到Redis"""
        r = self.client.client
        intent_data = {}

        for intent in self.routes:
            vector = self.embedding.embed_one(intent)
            intent_data[intent] = vector.tolist()

        key = f"{self.namespace}:data"
        r.delete(key)
        if intent_data:
            r.hset(key, mapping=intent_data)
            r.sadd(self._make_set_key(), *self.routes.keys())

        return True

    def load(self) -> bool:
        """从Redis加载路由"""
        r = self.client.client
        key = f"{self.namespace}:data"

        data = r.hgetall(key)
        if data:
            for intent, vec_str in data.items():
                self.intent_vectors[intent] = np.array(json.loads(vec_str), dtype=np.float32)
        return True

    def dispatch(self, query: str) -> Tuple[Any, str]:
        """
        路由分发

        Args:
            query: 用户查询

        Returns:
            (处理结果, 匹配的意图名)
        """
        query_vector = self.embedding.embed_one(query)
        query_vector = query_vector / np.linalg.norm(query_vector)

        best_intent = None
        best_score = self.threshold

        for intent, handler in self.routes.items():
            intent_vector = self.intent_vectors.get(intent)
            if intent_vector is None:
                intent_vector = self.embedding.embed_one(intent)
                self.intent_vectors[intent] = intent_vector
                intent_vector = intent_vector / np.linalg.norm(intent_vector)

            score = float(np.dot(query_vector, intent_vector))

            if score > best_score:
                best_score = score
                best_intent = intent

        if best_intent:
            return self.routes[best_intent](query), best_intent

        return self.default_handler(query), "default"

    def add_route(self, intent: str, handler: RouteHandler) -> "SemanticRouter":
        """添加路由（链式调用）"""
        return self.register(intent, handler)

    def remove_route(self, intent: str) -> bool:
        """移除路由"""
        if intent in self.routes:
            del self.routes[intent]
        if intent in self.intent_vectors:
            del self.intent_vectors[intent]
        return True

    def list_routes(self) -> List[str]:
        """列出所有路由"""
        return list(self.routes.keys())

    def get_handler(self, intent: str) -> Optional[RouteHandler]:
        """获取处理器"""
        return self.routes.get(intent)