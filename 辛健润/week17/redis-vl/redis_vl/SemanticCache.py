"""缓存模块 - 语义缓存 & 嵌入缓存"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import redis

from redis_vl.client import get_redis_client, RedisClient
from redis_vl.EmbeddingsCache import DashScopeEmbedding


class EmbeddingCache:
    """
    嵌入缓存 - 缓存文本到向量的转换结果

    使用示例:
        cache = EmbeddingCache()

        # 缓存embedding
        cache.set("hello world")

        # 获取缓存的embedding
        vector = cache.get("hello world")
    """

    def __init__(
        self,
        namespace: str = "emb_cache",
        ttl: int = 86400 * 30,  # 默认30天
        embedding_model: Optional[DashScopeEmbedding] = None,
        client: Optional[RedisClient] = None,
    ):
        self.namespace = namespace
        self.ttl = ttl
        self.embedding = embedding_model or DashScopeEmbedding()
        self.client = client or get_redis_client()

    def _make_key(self, text: str) -> str:
        """生成缓存键（基于文本hash）"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.namespace}:emb:{text_hash}"

    def _make_set_key(self) -> str:
        """生成集合键"""
        return f"{self.namespace}:sets"

    def set(self, text: str) -> Optional[np.ndarray]:
        """
        缓存文本的embedding

        Args:
            text: 文本

        Returns:
            embedding向量，如果出错返回None
        """
        key = self._make_key(text)

        try:
            vector = self.embedding.embed_one(text)
        except Exception:
            return None

        r = self.client.client
        pipe = r.pipeline()
        pipe.hset(key, mapping={"vector": json.dumps(vector.tolist())})
        pipe.expire(key, self.ttl)
        pipe.sadd(self._make_set_key(), key)
        pipe.execute()

        return vector

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        获取缓存的embedding

        Args:
            text: 文本

        Returns:
            embedding向量，如果未命中返回None
        """
        key = self._make_key(text)
        r = self.client.client

        data = r.hget(key, "vector")
        if data:
            return np.array(json.loads(data), dtype=np.float32)

        return None

    def get_or_compute(self, text: str) -> np.ndarray:
        """
        获取缓存的embedding，如果没有则计算并缓存

        Args:
            text: 文本

        Returns:
            embedding向量
        """
        vector = self.get(text)
        if vector is not None:
            return vector
        return self.set(text)

    def mget(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """批量获取"""
        return [self.get(t) for t in texts]

    def mget_or_compute(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取，未命中的会计算并缓存"""
        vectors = []
        for text in texts:
            vectors.append(self.get_or_compute(text))
        return vectors

    def delete(self, text: str) -> bool:
        """删除缓存"""
        key = self._make_key(text)
        r = self.client.client
        r.srem(self._make_set_key(), key)
        r.delete(key)
        return True

    def clear(self) -> bool:
        """清空缓存"""
        r = self.client.client
        keys = r.smembers(self._make_set_key())
        if keys:
            r.delete(*keys)
            r.delete(self._make_set_key())
        return True

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        r = self.client.client
        set_key = self._make_set_key()
        size = r.scard(set_key)
        return {
            "size": size,
            "ttl": self.ttl,
        }


class SemanticCache:
    """
    语义缓存 - 基于向量相似度的请求-结果缓存

    使用示例:
        cache = SemanticCache(threshold=0.85)

        # 缓存结果
        cache.set("今天天气好吗", "天气很好，阳光明媚！")

        # 获取缓存结果 - 相似问题会命中缓存
        result = cache.get("今天天气怎么样")
    """

    def __init__(
        self,
        namespace: str = "sem_cache",
        threshold: float = 0.85,
        ttl: int = 86400 * 7,  # 默认7天
        embedding_model: Optional[DashScopeEmbedding] = None,
        client: Optional[RedisClient] = None,
    ):
        self.namespace = namespace
        self.threshold = threshold
        self.ttl = ttl
        self.embedding = embedding_model or DashScopeEmbedding()
        self.client = client or get_redis_client()

    def _norm_vector(self, vec: np.ndarray) -> np.ndarray:
        """归一化向量（用于余弦相似度）"""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _make_key(self, prompt: str) -> str:
        """生成缓存键"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{self.namespace}:prompt:{prompt_hash}"

    def _make_set_key(self) -> str:
        """生成集合键"""
        return f"{self.namespace}:sets"

    def _calc_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """计算余弦相似度"""
        vec1 = self._norm_vector(vec1)
        vec2 = self._norm_vector(vec2)
        return float(np.dot(vec1, vec2))

    def set(self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        缓存问答对

        Args:
            prompt: 用户提问
            response: LLM回答
            metadata: 额外元数据

        Returns:
            是否成功
        """
        prompt_key = self._make_key(prompt)
        prompt_norm = self.embedding.embed_one(prompt)

        r = self.client.client
        pipe = r.pipeline()

        data = {
            "response": response,
            "vector": json.dumps(prompt_norm.tolist()),
            "timestamp": str(int(np.datetime64("now").astype(int))),
        }
        if metadata:
            data["metadata"] = json.dumps(metadata)

        pipe.hset(prompt_key, mapping=data)
        pipe.expire(prompt_key, self.ttl)
        pipe.sadd(self._make_set_key(), prompt_key)

        pipe.execute()
        return True

    def get(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        """
        获取缓存的问答对

        Args:
            prompt: 当前用户提问

        Returns:
            (缓存的回答, 相似度分数)
        """
        prompt_vector = self.embedding.embed_one(prompt)
        prompt_vector = self._norm_vector(prompt_vector)

        r = self.client.client
        set_key = self._make_set_key()

        candidates = r.smembers(set_key)
        if not candidates:
            return None, 0.0

        best_response = None
        best_score = 0.0

        for prompt_key in candidates:
            cached = r.hgetall(prompt_key)
            if not cached:
                continue

            cached_vector = np.array(json.loads(cached.get("vector", "[]")) or [], dtype=np.float32)
            if cached_vector.size == 0:
                continue

            similarity = self._calc_similarity(prompt_vector, cached_vector)

            if similarity > best_score and similarity >= self.threshold:
                best_score = similarity
                best_response = cached.get("response")

        return best_response, best_score

    def mget(self, prompts: List[str]) -> List[Tuple[Optional[str], Optional[float]]]:
        """批量获取缓存"""
        return [self.get(p) for p in prompts]

    def delete(self, prompt: str) -> bool:
        """删除缓存"""
        prompt_key = self._make_key(prompt)
        r = self.client.client
        r.srem(self._make_set_key(), prompt_key)
        r.delete(prompt_key)
        return True

    def clear(self) -> bool:
        """清空缓存"""
        r = self.client.client
        candidates = r.smembers(self._make_set_key())
        if candidates:
            r.delete(*candidates)
            r.delete(self._make_set_key())
        return True

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        r = self.client.client
        set_key = self._make_set_key()
        size = r.scard(set_key)
        return {
            "size": size,
            "threshold": self.threshold,
            "ttl": self.ttl,
        }