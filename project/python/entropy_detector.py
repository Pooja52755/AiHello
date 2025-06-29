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
    def __init__(self):
        self.entropy_threshold = 0.7
        self.max_traversal_depth = 5
        self.min_entropy = 0.1
        self.max_entropy = 0.9
        self.entropy_window = 3  # Window size for entropy smoothing
        self.gnn_embedding = GNNNodeEmbedding()
        self.blt_model = BLTEntropyModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache for node embeddings and entropies
        self.node_embeddings = {}
        self.edge_entropies = {}
        self._init_models()
        
    def _init_models(self):
        """Initialize models with default weights"""
        # In a real implementation, you would load pre-trained weights here
        pass
    
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
            return {
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'boundary_precision': 0.0,
                'traversal_efficiency': 0.0
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