"""对话历史管理 - SemanticMessageHistory"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import numpy as np
import redis
from redis_vl.client import get_redis_client, RedisClient
from redis_vl.EmbeddingsCache import DashScopeEmbedding


@dataclass
class Message:
    """对话消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(role=data.get("role", "user"), content=data.get("content", ""), name=data.get("name"))


class ConversationContext:
    """单次对话上下文"""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> None:
        """添加消息"""
        self.messages.append(Message(role=role, content=content, name=name))

    def to_llm_format(self) -> List[Dict[str, Any]]:
        """转换为LLM API格式"""
        return [msg.to_dict() for msg in self.messages]

    def to_context_text(self, include_recent: int = 4) -> str:
        """转换为上下文字符串"""
        recent = self.messages[-include_recent:] if len(self.messages) > include_recent else self.messages
        parts = []
        for msg in recent:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.messages)


class SemanticMessageHistory:
    """
    对话历史管理器 - 支持语义检索的对话历史

    使用示例:
        history = SemanticMessageHistory(session_id="user123")

        # 保存对话
        history.add_user_message("你好")
        history.add_assistant_message("你好，我是AI助手")
        history.add_user_message("今天天气怎么样")
        history.add_assistant_message("今天天气很好")

        # 获取历史（最近N轮）
        messages = history.get_messages(num_turns=2)

        # 语义检索相关历史
        history.save()
        related = history.search("天气"))
    """

    def __init__(
        self,
        session_id: str,
        namespace: str = "msg_history",
        max_turns: int = 20,
        ttl: int = 86400 * 7,
        embedding_model: Optional[DashScopeEmbedding] = None,
        client: Optional[RedisClient] = None,
    ):
        self.session_id = session_id
        self.namespace = namespace
        self.max_turns = max_turns
        self.ttl = ttl
        self.embedding = embedding_model or DashScopeEmbedding()
        self.client = client or get_redis_client()

        self.messages: List[Message] = []
        self._load()

    def _history_key(self) -> str:
        """历史记录key"""
        return f"{self.namespace}:history:{self.session_id}"

    def _index_key(self) -> str:
        """索引key"""
        return f"{self.namespace}:idx:{self.session_id}"

    def _load(self) -> bool:
        """加载历史"""
        r = self.client.client

        data = r.lrange(self._history_key(), 0, -1)
        if not data:
            return False

        self.messages = []
        for item in data:
            msg_dict = json.loads(item)
            self.messages.append(Message.from_dict(msg_dict))
        return True

    def add_user_message(self, content: str, name: Optional[str] = None) -> "SemanticMessageHistory":
        """添加用户消息"""
        self.messages.append(Message(role="user", content=content, name=name))
        return self

    def add_assistant_message(self, content: str, name: Optional[str] = None) -> "SemanticMessageHistory":
        """添加助手消息"""
        self.messages.append(Message(role="assistant", content=content, name=name))
        return self

    def add_system_message(self, content: str, name: Optional[str] = None) -> "SemanticMessageHistory":
        """添加系统消息"""
        self.messages.append(Message(role="system", content=content, name=name))
        return self

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> "SemanticMessageHistory":
        """添加任意角色消息"""
        self.messages.append(Message(role=role, content=content, name=name))
        return self

    def get_messages(self, num_turns: Optional[int] = None) -> List[Message]:
        """获取历史消息"""
        if num_turns:
            return self.messages[-num_turns * 2:]
        return self.messages.copy()

    def to_llm_format(self, num_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取LLM格式的消息列表"""
        messages = self.get_messages(num_turns)
        return [msg.to_dict() for msg in messages]

    def truncate(self, turns: int) -> "SemanticMessageHistory":
        """截断历史到指定轮数"""
        if len(self.messages) > turns * 2:
            self.messages = self.messages[-(turns * 2):]
        return self

    def save(self) -> bool:
        """保存历史到Redis"""
        r = self.client.client
        key = self._history_key()

        r.delete(key)
        if self.messages:
            items = [json.dumps(msg.to_dict()) for msg in self.messages]
            r.rpush(key, *items)
            r.expire(key, self.ttl)

        return True

    def clear(self) -> bool:
        """清空历史"""
        r = self.client.client
        r.delete(self._history_key())
        r.delete(self._index_key())
        self.messages = []
        return True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        语义检索相关历史

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相关消息列表
        """
        query_vector = self.embedding.embed_one(query)

        r = self.client.client
        results = r.ft(self._index_key()).knn_search(
            top_k,
            "vector",
            query_vector.tolist(),
            return_fields=["content", "role", "timestamp"],
        )

        return [
            {"content": doc.content, "role": doc.role, "score": doc.score}
            for doc in results.docs
        ]

    def rebuild_index(self) -> bool:
        """重建语义索引"""
        r = self.client.client

        index_name = self._index_key()
        try:
            r.ft(index_name).create_index(
                [redis.TextField("content"), redis.VectorField("vector", "HNSW", {"TYPE": "FLOAT32", "DIM": 1024, "DISTANCE_METRIC": "COSINE"})]
            )
        except Exception:
            pass

        pipe = r.pipeline()
        for i, msg in enumerate(self.messages):
            if msg.role == "user":
                vector = self.embedding.embed_one(msg.content)
                key = f"{index_name}:msg:{i}"
                pipe.hset(key, mapping={"content": msg.content, "role": msg.role, "vector": vector.tolist(), "timestamp": str(i)})

        pipe.execute()
        return True

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"<SemanticMessageHistory session_id={self.session_id} messages={len(self.messages)}>"