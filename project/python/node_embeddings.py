import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class NodeEmbeddingGenerator:
    """Generate and manage node embeddings for knowledge graph nodes"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', embedding_dim=384):
        """
        Initialize the node embedding generator
        
        Args:
            model_name: SentenceTransformer model name
            embedding_dim: Dimension of embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.embeddings_cache = {}
        
    def load_model(self):
        """Load the SentenceTransformer model"""
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def generate_node_embeddings(self, graph_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Generate embeddings for all nodes in the graph
        
        Args:
            graph_data: Graph data containing nodes and edges
            
        Returns:
            Dictionary mapping node IDs to their embedding vectors
        """
        self.load_model()
        
        node_embeddings = {}
        nodes = graph_data.get('nodes', [])
        
        # Extract node labels for embedding generation
        node_labels = []
        node_ids = []
        
        for node in nodes:
            node_id = node['id']
            node_label = node.get('label', node_id)
            
            node_labels.append(node_label)
            node_ids.append(node_id)
        
        if not node_labels:
            logger.warning("No nodes found for embedding generation")
            return {}
        
        logger.info(f"Generating embeddings for {len(node_labels)} nodes")
        
        # Generate embeddings in batch for efficiency
        try:
            embeddings = self.model.encode(node_labels, convert_to_tensor=False)
            
            # Convert to dictionary format
            for i, node_id in enumerate(node_ids):
                node_embeddings[node_id] = embeddings[i].tolist()
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to random embeddings
            for node_id in node_ids:
                node_embeddings[node_id] = np.random.normal(0, 1, self.embedding_dim).tolist()
        
        logger.info(f"Generated embeddings for {len(node_embeddings)} nodes")
        return node_embeddings
    
    def save_embeddings(self, embeddings: Dict[str, List[float]], filepath: str):
        """Save embeddings to JSON file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(embeddings, f, indent=2)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, List[float]]:
        """Load embeddings from JSON file"""
        try:
            with open(filepath, 'r') as f:
                embeddings = json.load(f)
            logger.info(f"Loaded embeddings from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {}
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def calculate_entropy_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate entropy score from cosine similarity
        
        Args:
            embedding1: First node embedding
            embedding2: Second node embedding
            
        Returns:
            Entropy score (1 - cosine_similarity)
        """
        similarity = self.calculate_cosine_similarity(embedding1, embedding2)
        entropy = 1.0 - similarity
        return max(0.0, min(1.0, entropy))  # Clamp between 0 and 1
    
    def get_node_embedding(self, node_label: str) -> List[float]:
        """Get embedding for a single node label"""
        if node_label in self.embeddings_cache:
            return self.embeddings_cache[node_label]
        
        self.load_model()
        try:
            embedding = self.model.encode([node_label], convert_to_tensor=False)[0]
            self.embeddings_cache[node_label] = embedding.tolist()
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for '{node_label}': {e}")
            # Return random embedding as fallback
            embedding = np.random.normal(0, 1, self.embedding_dim).tolist()
            self.embeddings_cache[node_label] = embedding
            return embedding

# Global instance for use across the application
embedding_generator = NodeEmbeddingGenerator()
