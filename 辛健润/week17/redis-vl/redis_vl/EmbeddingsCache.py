"""阿里百炼Embedding模型"""

from typing import List, Union
import numpy as np
from redis_vl.config import get_config


class DashScopeEmbedding:
    """阿里百炼Embedding模型封装"""

    def __init__(self, model: str = None, api_key: str = None):
        config = get_config()
        self.model = model or config.dashscope.model
        self.api_key = api_key or config.dashscope.api_key
        self._client = None

    @property
    def client(self):
        """延迟初始化DashScope客户端"""
        if self._client is None:
            import dashscope
            dashscope.api_key = self.api_key
            self._client = dashscope.TextEmbedding
        return self._client

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        将文本转换为向量

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量 numpy数组
        """
        if isinstance(texts, str):
            texts = [texts]

        resp = self.client.call(
            model=self.model,
            input={"texts": texts}
        )

        if resp.status_code == 200:
            embeddings = [item["embedding"] for item in resp.output["results"]]
            if len(embeddings) == 1:
                return np.array(embeddings[0], dtype=np.float32)
            return np.array(embeddings, dtype=np.float32)
        else:
            raise Exception(f"Embedding call failed: {resp.code} - {resp.message}")

    def embed_one(self, text: str) -> np.ndarray:
        """将单个文本转换为向量"""
        return self.embed(text)

    def get_dimension(self) -> int:
        """获取向量维度"""
        config = get_config()
        return config.vector.dimension