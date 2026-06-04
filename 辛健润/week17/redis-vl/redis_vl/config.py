"""配置管理模块"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class RedisConfig(BaseModel):
    """Redis连接配置"""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: Optional[str] = Field(default=None)
    db: int = Field(default=0)
    decode_responses: bool = Field(default=True)

    def to_redis_url(self) -> str:
        """生成Redis连接URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class DashScopeConfig(BaseModel):
    """阿里百炼配置"""
    api_key: str = Field(default="")
    model: str = Field(default="text-embedding-v3")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY", "")


class VectorConfig(BaseModel):
    """向量检索配置"""
    index_type: str = Field(default="HNSW")  # HNSW, FLAT
    dimension: int = Field(default=1024)  # 阿里百炼v3是1024维
    m: int = Field(default=16)  # HNSW参数
    ef_construction: int = Field(default=200)  # HNSW参数
    ef_search: int = Field(default=200)  # HNSW搜索参数


class Config(BaseModel):
    """全局配置"""
    redis: RedisConfig = Field(default_factory=RedisConfig)
    dashscope: DashScopeConfig = Field(default_factory=DashScopeConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)

_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """设置全局配置"""
    global _config
    _config = config