import numpy as np
import networkx as nx
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, deque, Counter
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random

logger = logging.getLogger(__name__)

class GNNNodeEmbedding(nn.Module):
    """Simple GNN for node embedding"""
    def __init__(self, input_dim=100, hidden_dim=64, output_dim=32):
        super(GNNNodeEmbedding, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class BLTEntropyModel:
    """BLT-inspired entropy model for graph traversal"""
    def __init__(self, node_embedding_dim=32, hidden_dim=64):
        self.node_embedding_dim = node_embedding_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simple MLP for boundary prediction
        self.boundary_predictor = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
    def calculate_entropy(self, node_embeddings, edge_index):
        """Calculate entropy scores for edges based on node embeddings"""
        src, dst = edge_index
        edge_embeddings = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=1)
        return self.boundary_predictor(edge_embeddings).squeeze()

class EntropyDetector:
    """Enhanced Entropy Detector with Node2Vec-style embeddings for sentence boundary detection"""
    
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.node_embeddings = {}
        self.entropy_scores = {}
        self.boundary_predictions = {}
        
        # Additional attributes for enhanced functionality
        self.entropy_threshold = 0.7
        self.max_traversal_depth = 5
        self.min_entropy = 0.1
        self.max_entropy = 0.9
        self.entropy_window = 3  # Window size for entropy smoothing
        self.gnn_embedding = GNNNodeEmbedding()
        self.blt_model = BLTEntropyModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache for edge entropies
        self.edge_entropies = {}
        self._init_models()
        
    def _init_models(self):
        """Initialize models with default weights"""
        # In a real implementation, you would load pre-trained weights here
        pass
        
    def compute_node_embeddings(self, graph_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute embeddings for each node using sentence transformer"""
        node_embeddings = {}
        
        for node in graph_data['nodes']:
            node_id = node['id']
            node_label = node['label']
            
            # Generate embedding for node label
            embedding = self.embedding_model.encode([node_label])[0]
            node_embeddings[node_id] = embedding
            
        self.node_embeddings = node_embeddings
        return node_embeddings
    
    def calculate_entropy_scores(self, graph_data: Dict[str, Any], start_node: str = None) -> Dict[str, float]:
        """Calculate entropy scores for each node relative to a starting node"""
        
        # Compute embeddings if not already done
        if not self.node_embeddings:
            self.compute_node_embeddings(graph_data)
        
        # If no start node specified, use the first node
        if start_node is None and graph_data['nodes']:
            start_node = graph_data['nodes'][0]['id']
        
        if start_node not in self.node_embeddings:
            logger.warning(f"Start node {start_node} not found in embeddings")
            return {}
        
        start_embedding = self.node_embeddings[start_node]
        entropy_scores = {}
        
        # Calculate cosine distance as entropy measure
        for node_id, embedding in self.node_embeddings.items():
            # Cosine distance (1 - cosine similarity)
            cosine_sim = cosine_similarity([start_embedding], [embedding])[0][0]
            entropy_score = 1.0 - cosine_sim
            
            # Normalize to [0, 1]
            entropy_score = max(0.0, min(1.0, entropy_score))
            entropy_scores[node_id] = float(entropy_score)
        
        self.entropy_scores = entropy_scores
        return entropy_scores
    
    def calculate_neighborhood_entropy(self, graph_data: Dict[str, Any], node_id: str) -> float:
        """Calculate entropy based on neighborhood embedding mean divergence"""
        
        if not self.node_embeddings:
            self.compute_node_embeddings(graph_data)
        
        if node_id not in self.node_embeddings:
            return 0.0
        
        # Build adjacency for quick neighbor lookup
        adjacency = defaultdict(list)
        for edge in graph_data['edges']:
            adjacency[edge['source']].append(edge['target'])
            adjacency[edge['target']].append(edge['source'])
        
        neighbors = adjacency[node_id]
        if not neighbors:
            return 0.0
        
        # Calculate neighborhood mean embedding
        neighbor_embeddings = []
        for neighbor in neighbors:
            if neighbor in self.node_embeddings:
                neighbor_embeddings.append(self.node_embeddings[neighbor])
        
        if not neighbor_embeddings:
            return 0.0
        
        neighborhood_mean = np.mean(neighbor_embeddings, axis=0)
        node_embedding = self.node_embeddings[node_id]
        
        # Calculate divergence (cosine distance)
        divergence = 1.0 - cosine_similarity([node_embedding], [neighborhood_mean])[0][0]
        return max(0.0, min(1.0, float(divergence)))
    
    def entropy_guided_traversal(self, graph_data: Dict[str, Any], start_node: str, 
                                entropy_threshold: float = 0.4) -> Dict[str, Any]:
        """Perform entropy-guided traversal with boundary detection"""
        
        # Calculate entropy scores
        entropy_scores = self.calculate_entropy_scores(graph_data, start_node)
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in graph_data['edges']:
            adjacency[edge['source']].append(edge['target'])
            adjacency[edge['target']].append(edge['source'])
        
        # Traversal state
        visited = set()
        boundary_nodes = set()
        traversal_path = []
        same_sentence_nodes = set()
        
        # BFS with entropy-based stopping
        queue = deque([start_node])
        visited.add(start_node)
        same_sentence_nodes.add(start_node)
        traversal_path.append(start_node)
        
        while queue:
            current_node = queue.popleft()
            
            # Explore neighbors
            for neighbor in adjacency[current_node]:
                if neighbor not in visited:
                    neighbor_entropy = entropy_scores.get(neighbor, 0.0)
                    
                    # Check if entropy threshold is crossed
                    if neighbor_entropy > entropy_threshold:
                        boundary_nodes.add(neighbor)
                        # Mark as boundary but don't continue traversal from here
                        visited.add(neighbor)
                        traversal_path.append(neighbor)
                    else:
                        # Continue traversal
                        visited.add(neighbor)
                        same_sentence_nodes.add(neighbor)
                        queue.append(neighbor)
                        traversal_path.append(neighbor)
        
        # Calculate evaluation metrics
        total_nodes = len(graph_data['nodes'])
        visited_count = len(same_sentence_nodes)
        boundary_count = len(boundary_nodes)
        efficiency = visited_count / total_nodes if total_nodes > 0 else 0.0
        
        # Update boundary predictions
        self.boundary_predictions = {
            node['id']: node['id'] in boundary_nodes 
            for node in graph_data['nodes']
        }
        
        return {
            'traversal_path': traversal_path,
            'visited_nodes': list(same_sentence_nodes),
            'boundary_nodes': list(boundary_nodes),
            'entropy_scores': entropy_scores,
            'boundary_predictions': self.boundary_predictions,
            'metrics': {
                'total_nodes': total_nodes,
                'visited_count': visited_count,
                'boundary_count': boundary_count,
                'efficiency': efficiency,
                'entropy_threshold': entropy_threshold
            }
        }
    
    def calculate_evaluation_metrics(self, predicted_boundaries: Set[str], 
                                   true_boundaries: Set[str] = None) -> Dict[str, float]:
        """Calculate evaluation metrics for boundary detection"""
        
        if true_boundaries is None:
            # Generate dummy ground truth for demonstration
            all_nodes = set(self.node_embeddings.keys())
            # Assume 20% of nodes are true boundaries (random for demo)
            true_boundaries = set(random.sample(list(all_nodes), 
                                              max(1, len(all_nodes) // 5)))
        
        all_nodes = set(self.node_embeddings.keys())
        
        # Convert to binary predictions
        y_true = [1 if node in true_boundaries else 0 for node in all_nodes]
        y_pred = [1 if node in predicted_boundaries else 0 for node in all_nodes]
        
        # Calculate metrics
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': len(predicted_boundaries.intersection(true_boundaries)),
            'false_positives': len(predicted_boundaries - true_boundaries),
            'false_negatives': len(true_boundaries - predicted_boundaries),
            'accuracy': sum(1 for i, pred in enumerate(y_pred) if pred == y_true[i]) / len(y_true)
        }
    
    def calculate_initial_entropy(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate initial entropy data for the knowledge graph"""
        
        if not kg_data or 'graph' not in kg_data:
            return {}
        
        graph_data = kg_data['graph']
        
        # Compute node embeddings
        self.compute_node_embeddings(graph_data)
        
        # Calculate entropy scores for all nodes (using first node as reference)
        if graph_data['nodes']:
            start_node = graph_data['nodes'][0]['id']
            entropy_scores = self.calculate_entropy_scores(graph_data, start_node)
        else:
            entropy_scores = {}
        
        # Calculate neighborhood entropies
        neighborhood_entropies = {}
        for node in graph_data['nodes']:
            node_id = node['id']
            neighborhood_entropies[node_id] = self.calculate_neighborhood_entropy(graph_data, node_id)
        
        return {
            'node_entropy_scores': entropy_scores,
            'neighborhood_entropies': neighborhood_entropies,
            'mean_entropy': np.mean(list(entropy_scores.values())) if entropy_scores else 0.0,
            'std_entropy': np.std(list(entropy_scores.values())) if entropy_scores else 0.0,
            'has_embeddings': len(self.node_embeddings) > 0
        }
    
    def calculate_initial_entropy(self, graph_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate initial entropy values for all nodes"""
        entropy_values = {}
        nodes = {node['id']: node for node in graph_data.get('nodes', [])}
        edges = graph_data.get('edges', [])
        
        # Build adjacency information
        adjacency = defaultdict(list)
        for edge in edges:
            adjacency[edge['source']].append(edge['target'])
            adjacency[edge['target']].append(edge['source'])
        
        for node_id, node_data in nodes.items():
            entropy_values[node_id] = self._calculate_node_entropy(
                node_id, node_data, adjacency, nodes
            )
        
        return entropy_values
    
    def _calculate_node_entropy(self, node_id: str, node_data: Dict, 
                               adjacency: Dict, all_nodes: Dict) -> float:
        """Calculate entropy for a single node using BLT-inspired approach"""
        
        # If we have precomputed entropy, use it
        if node_id in self.node_embeddings and 'entropy' in self.node_embeddings[node_id]:
            return self.node_embeddings[node_id]['entropy']
            
        # Base entropy from node characteristics
        base_entropy = self.min_entropy
        
        # 1. Sentence distribution entropy (BLT-inspired)
        sentence_ids = node_data.get('sentence_ids', [])
        if len(sentence_ids) > 1:
            # Node appears in multiple sentences - higher entropy (BLT: patch ambiguity)
            base_entropy += 0.2 * len(sentence_ids)
        
        # 2. Degree-based entropy (BLT: local context complexity)
        neighbors = adjacency.get(node_id, [])
        degree = len(neighbors)
        
        if degree == 0:
            base_entropy = self.max_entropy  # Isolated nodes have max entropy
        else:
            # Calculate neighbor sentence diversity (BLT: context diversity)
            neighbor_sentences = set()
            neighbor_degrees = []
            
            for neighbor_id in neighbors:
                neighbor_data = all_nodes.get(neighbor_id, {})
                neighbor_sentences.update(neighbor_data.get('sentence_ids', []))
                neighbor_degrees.append(len(adjacency.get(neighbor_id, [])))
            
            # BLT: Higher entropy for nodes with diverse contexts
            sentence_diversity = len(neighbor_sentences) / (len(sentence_ids) + 1e-6)
            degree_variation = np.std(neighbor_degrees) / (np.mean(neighbor_degrees) + 1e-6)
            
            base_entropy += 0.2 * min(sentence_diversity, 1.0)
            base_entropy += 0.1 * min(degree_variation, 1.0)
        
        # 3. Centrality-based entropy (BLT: structural importance)
        betweenness = node_data.get('betweenness', 0)
        if betweenness > 0.1:  # High betweenness indicates boundary node
            base_entropy = min(base_entropy + 0.3, self.max_entropy)
        
        # 4. Semantic coherence entropy (BLT: token prediction uncertainty)
        word_length = len(node_id.split())
        if word_length > 2:  # Complex phrases may indicate boundaries
            base_entropy = min(base_entropy + 0.15, self.max_entropy)
        
        # Store the calculated entropy
        if node_id not in self.node_embeddings:
            self.node_embeddings[node_id] = {}
        self.node_embeddings[node_id]['entropy'] = base_entropy
            
        return base_entropy
    
    def traverse_with_entropy(self, graph_data: Dict[str, Any], 
                             starting_nodes: List[str], 
                             threshold: float = None) -> Dict[str, Any]:
        """Perform entropy-based graph traversal"""
        
        if threshold is None:
            threshold = self.entropy_threshold
        
        # Calculate entropy values
        entropy_values = self.calculate_initial_entropy(graph_data)
        
        # Build adjacency list
        adjacency = defaultdict(list)
        edges = graph_data.get('edges', [])
        for edge in edges:
            adjacency[edge['source']].append(edge['target'])
            adjacency[edge['target']].append(edge['source'])
        
        results = []
        
        for start_node in starting_nodes:
            if start_node not in entropy_values:
                continue
            
            # Perform BFS traversal with entropy stopping condition
            traversal_result = self._entropy_bfs_traversal(
                start_node, adjacency, entropy_values, threshold
            )
            
            results.append({
                'starting_node': start_node,
                'visited_nodes': traversal_result['visited'],
                'boundary_nodes': traversal_result['boundaries'],
                'entropy_path': traversal_result['entropy_path'],
                'traversal_steps': traversal_result['steps'],
                'stopping_reason': traversal_result['stopping_reason']
            })
        
        return {
            'traversal_results': results,
            'entropy_values': entropy_values,
            'threshold_used': threshold,
            'summary': {
                'total_traversals': len(results),
                'avg_nodes_visited': np.mean([len(r['visited_nodes']) for r in results]) if results else 0,
                'avg_boundary_nodes': np.mean([len(r['boundary_nodes']) for r in results]) if results else 0
            }
        }
    
    def _entropy_bfs_traversal(self, start_node: str, adjacency: Dict, 
                              entropy_values: Dict[str, float], 
                              threshold: float) -> Dict[str, Any]:
        """Enhanced BFS traversal with BLT-inspired entropy-based stopping"""
        
        visited = set()
        boundary_nodes = set()
        entropy_path = []
        steps = []
        node_depth = {start_node: 0}
        parent_map = {}
        
        # Use a priority queue based on entropy (lower entropy first)
        queue = [(entropy_values.get(start_node, 0.5), 0, start_node)]  # (entropy, depth, node)
        
        stopping_reason = "max_depth_reached"
        entropy_window = []
        
        while queue and len(visited) < 100:  # Increased safety limit
            # Sort by entropy (lowest first) to explore more certain paths first
            queue.sort()
            current_entropy, depth, current_node = queue.pop(0)
            
            if current_node in visited or depth > self.max_traversal_depth:
                if depth > self.max_traversal_depth and current_node not in visited:
                    boundary_nodes.add(current_node)
                    stopping_reason = "max_depth_reached"
                continue
            
            # Add to visited and record path
            visited.add(current_node)
            current_entropy = entropy_values.get(current_node, 0.5)
            
            # Track entropy window for smoothing
            entropy_window.append(current_entropy)
            if len(entropy_window) > self.entropy_window:
                entropy_window.pop(0)
            
            # Use smoothed entropy
            smoothed_entropy = sum(entropy_window) / len(entropy_window)
            
            entropy_path.append({
                'node': current_node,
                'entropy': current_entropy,
                'smoothed_entropy': smoothed_entropy,
                'depth': depth,
                'is_boundary': False
            })
            
            steps.append({
                'step': len(steps),
                'node': current_node,
                'action': 'visit',
                'entropy': current_entropy,
                'smoothed_entropy': smoothed_entropy,
                'depth': depth
            })
            
            # BLT-inspired boundary detection
            is_boundary = False
            boundary_confidence = 0.0
            
            # 1. Check entropy threshold
            if smoothed_entropy > threshold and depth > 0:
                boundary_confidence += 0.5
                is_boundary = True
            
            # 2. Check entropy trend (increasing entropy suggests boundary)
            if len(entropy_path) >= 2:
                prev_entropy = entropy_path[-2]['smoothed_entropy']
                if smoothed_entropy > prev_entropy:
                    boundary_confidence += 0.3
                    is_boundary = True
            
            # 3. Check if this node connects to multiple sentences
            if len(entropy_path) > 1 and entropy_path[-1].get('sentence_id', -1) != entropy_path[-2].get('sentence_id', -2):
                boundary_confidence += 0.2
                is_boundary = True
            
            # If boundary confidence is high enough, mark as boundary
            if boundary_confidence >= 0.7:
                boundary_nodes.add(current_node)
                entropy_path[-1]['is_boundary'] = True
                stopping_reason = f"boundary_detected (confidence: {boundary_confidence:.2f})"
                
                steps.append({
                    'step': len(steps),
                    'node': current_node,
                    'action': 'boundary',
                    'entropy': current_entropy,
                    'smoothed_entropy': smoothed_entropy,
                    'confidence': boundary_confidence,
                    'reason': f"Boundary detected with confidence {boundary_confidence:.2f}"
                })
                
                # Optional: Don't explore beyond boundary nodes
                if boundary_confidence > 0.8:
                    continue
            
            # Add neighbors to queue with their entropy as priority
            neighbors = adjacency.get(current_node, [])
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in [n for _, _, n in queue]:
                    neighbor_entropy = entropy_values.get(neighbor, 0.5)
                    # Add some randomness to exploration
                    priority = neighbor_entropy + random.uniform(-0.1, 0.1)
                    queue.append((priority, depth + 1, neighbor))
                    node_depth[neighbor] = depth + 1
                    parent_map[neighbor] = current_node
                    
                    steps.append({
                        'step': len(steps),
                        'node': neighbor,
                        'action': 'queue',
                        'entropy': neighbor_entropy,
                        'depth': depth + 1,
                        'parent': current_node
                    })
        
        return {
            'visited': list(visited),
            'boundaries': list(boundary_nodes),
            'entropy_path': entropy_path,
            'steps': steps,
            'stopping_reason': stopping_reason
        }
    
    def evaluate_predictions(self, predictions: List[Dict], 
                           ground_truth: List[Dict]) -> Dict[str, float]:
        """Enhanced evaluation with BLT-inspired metrics"""
        
        if not predictions or not ground_truth:
            return {}
            
        # Convert ground truth to node sets
        gt_sets = [set(gt['nodes']) for gt in ground_truth if 'nodes' in gt]
        pred_sets = [set(pred.get('visited_nodes', [])) for pred in predictions]
        
        # Calculate precision, recall, F1 for boundary detection
        y_true = []
        y_pred = []
        
        # For each ground truth set, find best matching prediction
        for gt_set in gt_sets:
            best_f1 = 0
            best_pred = set()
            
            for pred_set in pred_sets:
                if not pred_set:
                    continue
                    
                # Calculate intersection and union
                intersection = len(gt_set.intersection(pred_set))
                union = len(gt_set.union(pred_set))
                
                # Calculate precision, recall, F1
                precision = intersection / len(pred_set) if pred_set else 0
                recall = intersection / len(gt_set) if gt_set else 0
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_pred = pred_set
            
            # Update true and predicted labels
            for node in gt_set:
                y_true.append(1 if node in best_pred else 0)
                y_pred.append(1 if node in gt_set else 0)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Additional BLT-inspired metrics
        avg_boundary_entropy = np.mean([
            p['entropy'] 
            for pred in predictions 
            for p in pred.get('entropy_path', []) 
            if p.get('is_boundary', False)
        ] or [0])
        
        traversal_efficiency = np.mean([
            len(pred.get('visited_nodes', [])) / (len(pred.get('boundary_nodes', [])) + 1e-6)
            for pred in predictions
        ] or [0])
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_boundary_entropy': float(avg_boundary_entropy),
            'traversal_efficiency': float(traversal_efficiency),
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth)
        }
        
        # Calculate F1 score for sentence groupings
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_boundary_precision = 0.0
        total_efficiency = 0.0
        
        for pred, truth in zip(predictions, ground_truth):
            pred_nodes = set(pred.get('visited_nodes', []))
            truth_nodes = set(truth.get('true_sentence_nodes', []))
            
            # Calculate precision, recall, F1
            if pred_nodes:
                precision = len(pred_nodes & truth_nodes) / len(pred_nodes)
                total_precision += precision
            
            if truth_nodes:
                recall = len(pred_nodes & truth_nodes) / len(truth_nodes)
                total_recall += recall
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                total_f1 += f1
            
            # Boundary precision
            pred_boundaries = set(pred.get('boundary_nodes', []))
            truth_boundaries = set(truth.get('true_boundary_nodes', []))
            
            if pred_boundaries:
                boundary_prec = len(pred_boundaries & truth_boundaries) / len(pred_boundaries)
                total_boundary_precision += boundary_prec
            
            # Traversal efficiency (fewer unnecessary visits is better)
            max_possible_nodes = len(truth_nodes)
            efficiency = max_possible_nodes / len(pred_nodes) if pred_nodes else 0
            total_efficiency += efficiency
        
        n = len(predictions)
        return {
            'f1_score': total_f1 / n,
            'precision': total_precision / n,
            'recall': total_recall / n,
            'boundary_precision': total_boundary_precision / n,
            'traversal_efficiency': total_efficiency / n,
            'num_evaluations': n
        }