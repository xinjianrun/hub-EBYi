"""单元测试"""

import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# 设置测试环境变量
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")


class TestEmbeddingsCache:
    """测试 EmbeddingsCache"""

    def test_make_key(self):
        """测试缓存key生成"""
        from redis_vl.SemanticCache import EmbeddingCache

        cache = EmbeddingCache(namespace="test")
        key1 = cache._make_key("hello")
        key2 = cache._make_key("hello")
        assert key1 == key2
        assert "test" in key1

    def test_make_set_key(self):
        """测试集合key生成"""
        from redis_vl.SemanticCache import EmbeddingCache

        cache = EmbeddingCache(namespace="test_emb")
        assert cache._make_set_key() == "test_emb:sets"

    def test_stats(self):
        """测试缓存统计"""
        from redis_vl.SemanticCache import EmbeddingCache

        cache = EmbeddingCache(ttl=3600)
        stats = cache.stats()
        assert "size" in stats
        assert "ttl" in stats
        assert stats["ttl"] == 3600


class TestSemanticCache:
    """测试 SemanticCache"""

    def test_make_key(self):
        """测试缓存key生成"""
        from redis_vl.SemanticCache import SemanticCache

        cache = SemanticCache(namespace="test")
        key1 = cache._make_key("你好")
        key2 = cache._make_key("你好")
        assert key1 == key2

    def test_threshold_default(self):
        """测试默认阈值"""
        from redis_vl import SemanticCache
        cache = SemanticCache()
        assert cache.threshold == 0.85

    def test_custom_threshold(self):
        """测试自定义阈值"""
        from redis_vl import SemanticCache
        cache = SemanticCache(threshold=0.9)
        assert cache.threshold == 0.9


class TestSemanticRouter:
    """测试 SemanticRouter"""

    def test_make_key(self):
        """测试key生成"""
        from redis_vl.SemanticRouter import SemanticRouter

        router = SemanticRouter(namespace="test")
        key = router._make_key("天气")
        assert "test" in key
        assert "天气" in key

    def test_make_set_key(self):
        """测试集合key"""
        from redis_vl.SemanticRouter import SemanticRouter

        router = SemanticRouter(namespace="test")
        assert router._make_set_key() == "test:routes"

    def test_register_decorator(self):
        """测试装饰器注册"""
        from redis_vl.SemanticRouter import SemanticRouter

        router = SemanticRouter()

        @router.route("测试意图")
        def handler(query):
            return "结果"

        assert "测试意图" in router.routes

    def test_add_route(self):
        """测试手动注册"""
        from redis_vl.SemanticRouter import SemanticRouter

        router = SemanticRouter()
        router.add_route("测试", lambda x: "result")
        assert "测试" in router.routes

    def test_remove_route(self):
        """测试移除路由"""
        from redis_vl.SemanticRouter import SemanticRouter

        router = SemanticRouter()
        router.add_route("测试", lambda x: "result")
        router.remove_route("测试")
        assert "测试" not in router.routes

    def test_list_routes(self):
        """测试列出路由"""
        from redis_vl.SemanticRouter import SemanticRouter

        router = SemanticRouter()
        router.add_route("A", lambda x: "a")
        router.add_route("B", lambda x: "b")
        routes = router.list_routes()
        assert "A" in routes
        assert "B" in routes


class TestSemanticMessageHistory:
    """测试 SemanticMessageHistory"""

    def test_message_to_dict(self):
        """测试消息转字典"""
        from redis_vl.SemanticMessageHistory import Message

        msg = Message(role="user", content="你好")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "你好"

    def test_message_from_dict(self):
        """测试从字典创建消息"""
        from redis_vl.SemanticMessageHistory import Message

        msg = Message.from_dict({"role": "assistant", "content": "你好"})
        assert msg.role == "assistant"
        assert msg.content == "你好"

    def test_add_user_message(self):
        """测试添加用户消息"""
        from redis_vl.SemanticMessageHistory import SemanticMessageHistory

        history = SemanticMessageHistory(session_id="test")
        history.add_user_message("hello")
        assert len(history.messages) == 1
        assert history.messages[0].role == "user"

    def test_add_assistant_message(self):
        """测试添加助手消息"""
        from redis_vl.SemanticMessageHistory import SemanticMessageHistory

        history = SemanticMessageHistory(session_id="test")
        history.add_assistant_message("hi")
        assert len(history.messages) == 1
        assert history.messages[0].role == "assistant"

    def test_get_messages(self):
        """测试获取消息"""
        from redis_vl.SemanticMessageHistory import SemanticMessageHistory

        history = SemanticMessageHistory(session_id="test")
        history.add_user_message("1")
        history.add_assistant_message("2")
        history.add_user_message("3")
        history.add_assistant_message("4")

        msgs = history.get_messages(num_turns=1)
        assert len(msgs) == 2

    def test_truncate(self):
        """测试截断"""
        from redis_vl.SemanticMessageHistory import SemanticMessageHistory

        history = SemanticMessageHistory(session_id="test")
        for i in range(10):
            history.add_user_message(str(i))
            history.add_assistant_message(str(i))

        history.truncate(2)
        assert len(history) == 4

    def test_to_llm_format(self):
        """测试转换为LLM格式"""
        from redis_vl.SemanticMessageHistory import SemanticMessageHistory

        history = SemanticMessageHistory(session_id="test")
        history.add_user_message("你好")
        history.add_assistant_message("你好，我是AI")

        fmt = history.to_llm_format()
        assert len(fmt) == 2
        assert fmt[0]["role"] == "user"


class TestHybridSearch:
    """测试 HybridSearch"""

    def test_init(self):
        """测试初始化"""
        from redis_vl.search import HybridSearch

        search = HybridSearch(index_name="test_idx")
        assert search.index_name == "test_idx"


class TestSchema:
    """测试 Schema"""

    def test_field_type(self):
        """测试字段类型"""
        from redis_vl.schema import FieldType

        assert FieldType.TEXT == "TEXT"
        assert FieldType.TAG == "TAG"
        assert FieldType.VECTOR == "VECTOR"

    def test_distance_metric(self):
        """测试距离度量"""
        from redis_vl.schema import DistanceMetric

        assert DistanceMetric.COSINE == "COSINE"
        assert DistanceMetric.L2 == "L2"
        assert DistanceMetric.IP == "IP"

    def test_index_field(self):
        """测试索引字段"""
        from redis_vl.schema import IndexField, FieldType

        field = IndexField(name="content", field_type=FieldType.TEXT)
        assert field.name == "content"
        assert field.field_type == FieldType.TEXT

    def test_vector_index(self):
        """测试向量索引"""
        from redis_vl.schema import VectorIndex, DistanceMetric

        vi = VectorIndex(distance_metric=DistanceMetric.L2, dimension=512)
        assert vi.dimension == 512
        assert vi.distance_metric == DistanceMetric.L2

    def test_index_schema(self):
        """测试索引Schema"""
        from redis_vl.schema import IndexSchema, IndexField, FieldType, VectorIndex

        schema = IndexSchema(
            index_name="test",
            prefix="test",
            fields=[IndexField(name="content", field_type=FieldType.TEXT)],
            vector_index=VectorIndex(dimension=1024),
        )
        assert schema.index_name == "test"
        assert schema.prefix == "test"
        assert schema.vector_index.dimension == 1024


class TestConfig:
    """测试配置"""

    def test_redis_config(self):
        """测试Redis配置"""
        from redis_vl.config import RedisConfig

        config = RedisConfig(host="localhost", port=6379)
        assert config.host == "localhost"
        assert config.port == 6379

    def test_redis_url(self):
        """测试Redis URL生成"""
        from redis_vl.config import RedisConfig

        config = RedisConfig(host="localhost", port=6379)
        assert "localhost" in config.to_redis_url()

    def test_dashscope_config(self):
        """测试百炼配置"""
        from redis_vl.config import DashScopeConfig

        config = DashScopeConfig(api_key="sk-test", model="text-embedding-v3")
        assert config.api_key == "sk-test"
        assert config.model == "text-embedding-v3"

    def test_vector_config(self):
        """测试向量配置"""
        from redis_vl.config import VectorConfig

        config = VectorConfig(dimension=1536)
        assert config.dimension == 1536

    def test_get_config(self):
        """测试获取配置"""
        from redis_vl.config import get_config

        cfg = get_config()
        assert cfg.redis is not None
        assert cfg.dashscope is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])