"""Redis客户端封装"""

import redis
from typing import Optional
from redis_vl.config import get_config, RedisConfig


class RedisClient:
    """Redis客户端封装"""

    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or get_config().redis
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        """获取Redis连接"""
        if self._client is None:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                decode_responses=self.config.decode_responses,
            )
        return self._client

    def close(self) -> None:
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 全局客户端
_client: Optional[RedisClient] = None


def get_redis_client(config: Optional[RedisConfig] = None) -> RedisClient:
    """获取Redis客户端单例"""
    global _client
    if _client is None:
        _client = RedisClient(config)
    return _client


def close_redis_client() -> None:
    """关闭全局Redis客户端"""
    global _client
    if _client:
        _client.close()
        _client = None