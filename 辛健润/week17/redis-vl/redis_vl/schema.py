"""Schema管理 - 统一向量数据管理"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import redis
from pydantic import BaseModel, Field

from redis_vl.client import get_redis_client, RedisClient
from redis_vl.config import get_config, VectorConfig


class FieldType(str, Enum):
    """字段类型"""
    TEXT = "TEXT"
    TAG = "TAG"
    NUMERIC = "NUMERIC"
    VECTOR = "VECTOR"
    GEOSHAPE = "GEOSHAPE"


class DistanceMetric(str, Enum):
    """距离度量方式"""
    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"  # 内积


class IndexField(BaseModel):
    """索引字段定义"""
    name: str
    field_type: FieldType
    sortable: bool = False
    no_index: bool = False


class VectorIndex(BaseModel):
    """向量索引配置"""
    algorithm: str = "HNSW"  # HNSW, FLAT
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    dimension: int = 1024
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 200
    initial_cap: int = 100000
    priority: int = 0


class IndexSchema(BaseModel):
    """索引Schema定义"""
    index_name: str
    prefix: str  # 用于key_PREFIX:* 的前缀
    vector_index: Optional[VectorIndex] = None
    fields: List[IndexField] = Field(default_factory=list)

    def to_ft_create_args(self) -> List[str]:
        """转换为FT.CREATE命令参数"""
        args = [self.prefix]

        if self.vector_index:
            vi = self.vector_index
            vector_args = {
                "algorithm": vi.algorithm,
                "dimension": vi.dimension,
                "distance_metric": vi.distance_metric.value,
                "m": vi.m,
                "ef_construction": vi.ef_construction,
                "ef_search": vi.ef_search,
                "initial_cap": vi.initial_cap,
                "priority": vi.priority,
            }
            args.extend([
                "ON", "HASH",
                "SCHEMA"
            ] + self._build_schema_args())
        else:
            args.extend(["ON", "HASH", "SCHEMA"])

        return args

    def _build_schema_args(self) -> List[str]:
        """构建Schema参数"""
        schema_args = []
        for field in self.fields:
            schema_args.append(field.name)
            schema_args.append(field.field_type.value)
            if field.sortable:
                schema_args.append("SORTABLE")
            if field.no_index:
                schema_args.append("NOINDEX")
        return schema_args

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


class SchemaManager:
    """Schema管理器"""

    def __init__(self, client: Optional[RedisClient] = None):
        self.client = client or get_redis_client()

    def create_index(self, schema: IndexSchema) -> bool:
        """创建索引"""
        r = self.client.client
        cmd = ["FT.CREATE", schema.index_name] + schema.to_ft_create_args()[1:]
        try:
            r.execute_command(*cmd)
            return True
        except redis.ResponseError as e:
            if "already exists" in str(e).lower():
                return True
            raise

    def drop_index(self, index_name: str, sync: bool = False) -> bool:
        """删除索引"""
        r = self.client.client
        cmd = ["FT.DROPINDEX", index_name]
        if sync:
            cmd.append("DD")
        r.execute_command(*cmd)
        return True

    def list_indexes(self) -> List[str]:
        """列出所有索引"""
        r = self.client.client
        result = r.execute_command("FT._LIST")
        return [idx for idx in result if idx != "_"] if result else []

    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """获取索引信息"""
        r = self.client.client
        info = r.ft(index_name).info()
        return dict(info)

    def exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        return index_name in self.list_indexes()