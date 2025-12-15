"""
그래프 임베딩
GNN 기반 노드/그래프 임베딩 모듈
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# PyTorch 및 PyTorch Geometric은 선택적 import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from .knowledge_graph import KnowledgeGraph


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    input_dim: int = 384
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    aggregation: str = "mean"  # mean, max, sum
    model_type: str = "GCN"  # GCN, GAT, GraphSAGE


class GNNEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    GNN 기반 그래프 인코더
    
    지원 모델:
    - GCN: Graph Convolutional Network
    - GAT: Graph Attention Network
    - GraphSAGE: Sample and Aggregate
    - grn: Graph relation network
    """
    
    def __init__(self, config: EmbeddingConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다: pip install torch")
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric이 필요합니다: pip install torch-geometric")
        
        super().__init__()
        self.config = config
        
        # 레이어 선택
        conv_class = {
            "GCN": GCNConv,
            "GAT": GATConv,
            "GraphSAGE": SAGEConv,
        }.get(config.model_type, GCNConv)
        
        # 컨볼루션 레이어
        self.convs = nn.ModuleList()
        self.convs.append(conv_class(config.input_dim, config.hidden_dim))
        
        for _ in range(config.num_layers - 2):
            self.convs.append(conv_class(config.hidden_dim, config.hidden_dim))
        
        self.convs.append(conv_class(config.hidden_dim, config.output_dim))
        
        # 드롭아웃
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: "torch.Tensor", 
        edge_index: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        순전파
        
        Args:
            x: 노드 특징 행렬 [num_nodes, input_dim]
            edge_index: 엣지 인덱스 [2, num_edges]
            
        Returns:
            노드 임베딩 [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def encode_graph(
        self, 
        x: "torch.Tensor", 
        edge_index: "torch.Tensor",
        batch: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """
        그래프 레벨 임베딩
        
        Args:
            x: 노드 특징 행렬
            edge_index: 엣지 인덱스
            batch: 배치 할당 벡터
            
        Returns:
            그래프 임베딩
        """
        node_embeddings = self.forward(x, edge_index)
        
        if batch is None:
            # 단일 그래프
            if self.config.aggregation == "mean":
                return node_embeddings.mean(dim=0, keepdim=True)
            elif self.config.aggregation == "max":
                return node_embeddings.max(dim=0, keepdim=True)[0]
            else:  # sum
                return node_embeddings.sum(dim=0, keepdim=True)
        else:
            # 배치 그래프
            from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
            
            pool_fn = {
                "mean": global_mean_pool,
                "max": global_max_pool,
                "sum": global_add_pool,
            }.get(self.config.aggregation, global_mean_pool)
            
            return pool_fn(node_embeddings, batch)


class GraphEmbedder:
    """
    지식 그래프 임베딩 관리자
    
    쿼리 임베딩과 컨텍스트 임베딩을 통한 그래프 정렬 수행
    """
    
    def __init__(
        self, 
        config: EmbeddingConfig = None,
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.config = config or EmbeddingConfig()
        self.text_encoder_name = text_encoder
        
        self.gnn_encoder: Optional[GNNEncoder] = None
        self.text_encoder = None
        
        self._initialized = False
    
    def initialize(self):
        """모델 초기화"""
        if self._initialized:
            return
        
        # GNN 인코더
        if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
            self.gnn_encoder = GNNEncoder(self.config)
        
        # 텍스트 인코더
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(self.text_encoder_name)
        except ImportError:
            print("Warning: sentence-transformers not available")
        
        self._initialized = True
    
    def encode_text(self, text: str) -> np.ndarray:
        """텍스트를 임베딩으로 변환"""
        if self.text_encoder is None:
            # 폴백: 랜덤 임베딩
            return np.random.randn(self.config.input_dim).astype(np.float32)
        
        return self.text_encoder.encode(text, convert_to_numpy=True)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """여러 텍스트를 임베딩으로 변환"""
        if self.text_encoder is None:
            return np.random.randn(len(texts), self.config.input_dim).astype(np.float32)
        
        return self.text_encoder.encode(texts, convert_to_numpy=True)
    
    def embed_knowledge_graph(
        self, 
        kg: KnowledgeGraph
    ) -> Tuple[np.ndarray, List[str]]:
        """
        지식 그래프 임베딩
        
        Args:
            kg: 지식 그래프
            
        Returns:
            (노드 임베딩, 노드 ID 리스트)
        """
        if not self._initialized:
            self.initialize()
        
        # 노드 특징 추출
        node_ids = list(kg.nodes.keys())
        
        # 텍스트 기반 초기 임베딩
        labels = [kg.nodes[nid].label for nid in node_ids]
        initial_features = self.encode_texts(labels)
        
        # GNN으로 구조 정보 반영
        if self.gnn_encoder is not None and len(kg.edges) > 0:
            edge_index, _ = kg.to_edge_index()
            
            x = torch.tensor(initial_features, dtype=torch.float32)
            edge_idx = torch.tensor(edge_index, dtype=torch.long)
            
            with torch.no_grad():
                node_embeddings = self.gnn_encoder(x, edge_idx)
                return node_embeddings.numpy(), node_ids
        
        return initial_features, node_ids
    
    def align_query_to_graph(
        self, 
        query: str, 
        kg: KnowledgeGraph,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        쿼리와 그래프 노드 정렬
        
        Args:
            query: 쿼리 텍스트
            kg: 지식 그래프
            top_k: 반환할 상위 노드 수
            
        Returns:
            [(노드 ID, 유사도 점수)] 리스트
        """
        if not self._initialized:
            self.initialize()
        
        # 쿼리 임베딩
        query_embedding = self.encode_text(query)
        
        # 그래프 임베딩
        node_embeddings, node_ids = self.embed_knowledge_graph(kg)
        
        # 코사인 유사도 계산
        similarities = self._cosine_similarity(
            query_embedding.reshape(1, -1), 
            node_embeddings
        )[0]
        
        # 상위 k개 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(node_ids[i], float(similarities[i])) for i in top_indices]
    
    def compute_context_alignment(
        self,
        context_embedding: np.ndarray,
        kg: KnowledgeGraph,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        컨텍스트와 그래프 정렬
        
        Args:
            context_embedding: 컨텍스트 임베딩
            kg: 지식 그래프
            threshold: 유사도 임계값
            
        Returns:
            임계값 이상의 [(노드 ID, 유사도)] 리스트
        """
        if not self._initialized:
            self.initialize()
        
        node_embeddings, node_ids = self.embed_knowledge_graph(kg)
        
        similarities = self._cosine_similarity(
            context_embedding.reshape(1, -1),
            node_embeddings
        )[0]
        
        results = [
            (node_ids[i], float(similarities[i]))
            for i in range(len(node_ids))
            if similarities[i] >= threshold
        ]
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _cosine_similarity(
        self, 
        a: np.ndarray, 
        b: np.ndarray
    ) -> np.ndarray:
        """코사인 유사도 계산"""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)

