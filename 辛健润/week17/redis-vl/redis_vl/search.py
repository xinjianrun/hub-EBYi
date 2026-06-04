"""混合查询引擎 - 向量检索+元数据过滤+关键词全文搜索"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import redis

from redis_vl.client import get_redis_client, RedisClient
from redis_vl.EmbeddingsCache import DashScopeEmbedding


@dataclass
class SearchResult:
    """检索结果"""
    id: str
    score: float
    payload: Dict[str, Any]


class HybridSearch:
    """混合查询引擎"""

    def __init__(
        self,
        index_name: str,
        embedding_model: Optional[DashScopeEmbedding] = None,
        client: Optional[RedisClient] = None,
    ):
        self.index_name = index_name
        self.embedding = embedding_model or DashScopeEmbedding()
        self.client = client or get_redis_client()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_fields: Optional[List[str]] = None,
        hybrid_ratio: float = 0.5,
    ) -> List[SearchResult]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件，如 {"category": "news"}
            return_fields: 返回字段
            hybrid_ratio: 向量权重ybrid_ratio 0.5表示向量和关键词各占一半

        Returns:
            检索结果列表
        """
        query_vector = self.embedding.embed_one(query)

        return self.vector_search(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            filters=filters,
            return_fields=return_fields,
            hybrid_ratio=hybrid_ratio,
        )

    def vector_search(
        self,
        query_vector: np.ndarray,
        query_text: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_fields: Optional[List[str]] = None,
        hybrid_ratio: float = 0.5,
    ) -> List[SearchResult]:
        """向量检索"""
        r = self.client.client
        ft = r.ft(self.index_name)

        vector_field = "vector"

        query_list = []
        if hybrid_ratio > 0:
            query_list.append(
                f"KNN {top_k} @{vector_field} AS score_vec "
                f"$vec KNN {top_k} @{vector_field} => {{$vec: $vec, $k: {top_k}}}"
            )
        if query_text and hybrid_ratio < 1:
            query_list.append(query_text) if not query_list else None

        if filters:
            filter_parts = []
            for k, v in filters.items():
                if isinstance(v, list):
                    filter_parts.append(f"@{k}:[{v[0]} {v[-1]}]")
                else:
                    filter_parts.append(f"@{k}:{v}")
            filter_str = " & ".join(filter_parts)
            if query_list:
                query_list = [f"({q}) ({filter_str})" for q in query_list]
            else:
                query_list = [filter_str]

        query_str = " | ".join(query_list) if query_list else "*"

        params = {"vec": np.array(query_vector).astype(np.float32).tobytes()}

        return_fields = return_fields or ["*"]

        try:
            results = ft.search(
                query_str,
                params=params,
                sortby="score" if hybrid_ratio > 0 else None,
                first=0,
                num=top_k,
                return_properties=return_fields,
            )
        except redis.ResponseError:
            results = ft.knn_search(
                top_k,
                vector_field,
                query_vector.tolist(),
                return_fields=return_fields,
            )

        return [
            SearchResult(
                id=doc.id,
                score=doc.score,
                payload={k: getattr(doc, k) for k in return_fields if hasattr(doc, k)},
            )
            for doc in results.docs
        ]

    def text_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_fields: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """纯文本检索（关键词搜索）"""
        r = self.client.client
        ft = r.ft(self.index_name)

        query_str = query
        if filters:
            filter_parts = []
            for k, v in filters.items():
                if isinstance(v, list):
                    filter_parts.append("@%s:[%s %s]" % (k, v[0], v[-1]))
                else:
                    filter_parts.append("@%s:%s" % (k, v))
            query_str += " " + " & ".join(filter_parts)

        return_fields = return_fields or ["*"]

        results = ft.search(
            query_str,
            first=0,
            num=top_k,
            return_properties=return_fields,
        )

        return [
            SearchResult(
                id=doc.id,
                score=doc.score,
                payload={k: getattr(doc, k) for k in return_fields if hasattr(doc, k)},
            )
            for doc in results.docs
        ]

    def add_document(
        self,
        doc_id: str,
        vector: np.ndarray,
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """添加文档"""
        r = self.client.client
        key = f"doc:{doc_id}"

        doc_data = {"vector": vector.tolist()}
        if payload:
            doc_data.update(payload)

        r.hset(key, mapping=doc_data)
        r.sadd(f"{self.index_name}:docs", key)
        return True

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """批量添加文档"""
        pipe = self.client.client.pipeline()

        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids else str(i)
            key = f"doc:{doc_id}"

            vec = doc.get("vector")
            if vec is not None:
                if isinstance(vec, list):
                    vec = np.array(vec, dtype=np.float32)
                else:
                    vec = vec.astype(np.float32)

                doc_data = {"vector": vec.tolist()}
                payload = {k: v for k, v in doc.items() if k != "vector"}
                doc_data.update(payload)

                pipe.hset(key, mapping=doc_data)
                pipe.sadd(f"{self.index_name}:docs", key)

        pipe.execute()
        return True

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        r = self.client.client
        key = f"doc:{doc_id}"
        r.delete(key)
        r.srem(f"{self.index_name}:docs", key)
        return True