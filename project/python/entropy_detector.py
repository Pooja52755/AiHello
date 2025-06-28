import numpy as np
import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, deque
import logging
import math

logger = logging.getLogger(__name__)

class EntropyDetector:
    def __init__(self):
        self.entropy_threshold = 0.7
        self.max_traversal_depth = 5
    
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
        
        # Base entropy from node characteristics
        base_entropy = 0.0
        
        # 1. Sentence distribution entropy
        sentence_ids = node_data.get('sentence_ids', [])
        if len(sentence_ids) > 1:
            # Node appears in multiple sentences - higher entropy
            base_entropy += 0.3
        
        # 2. Degree-based entropy
        neighbors = adjacency.get(node_id, [])
        degree = len(neighbors)
        
        if degree == 0:
            base_entropy += 0.5  # Isolated nodes have high entropy
        else:
            # Calculate neighbor sentence diversity
            neighbor_sentences = set()
            for neighbor_id in neighbors:
                neighbor_data = all_nodes.get(neighbor_id, {})
                neighbor_sentences.update(neighbor_data.get('sentence_ids', []))
            
            # If neighbors span multiple sentences, increase entropy
            if len(neighbor_sentences) > len(sentence_ids):
                base_entropy += 0.4
        
        # 3. Centrality-based entropy
        betweenness = node_data.get('betweenness', 0)
        if betweenness > 0.1:  # High betweenness indicates boundary node
            base_entropy += 0.3
        
        # 4. Semantic coherence entropy (simplified)
        # In a full implementation, this would use embeddings
        word_length = len(node_id.split())
        if word_length > 2:  # Complex phrases may indicate boundaries
            base_entropy += 0.2
        
        return min(base_entropy, 1.0)
    
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
        """BFS traversal with entropy-based stopping"""
        
        visited = set()
        boundary_nodes = set()
        entropy_path = []
        steps = []
        queue = deque([(start_node, 0)])  # (node, depth)
        
        stopping_reason = "max_depth_reached"
        
        while queue and len(visited) < 50:  # Safety limit
            current_node, depth = queue.popleft()
            
            if current_node in visited or depth > self.max_traversal_depth:
                continue
            
            visited.add(current_node)
            current_entropy = entropy_values.get(current_node, 0.5)
            
            entropy_path.append({
                'node': current_node,
                'entropy': current_entropy,
                'depth': depth
            })
            
            steps.append({
                'step': len(steps),
                'node': current_node,
                'action': 'visit',
                'entropy': current_entropy,
                'depth': depth
            })
            
            # Check if we should stop (high entropy indicates boundary)
            if current_entropy > threshold and depth > 0:
                boundary_nodes.add(current_node)
                stopping_reason = "entropy_threshold_reached"
                steps.append({
                    'step': len(steps),
                    'node': current_node,
                    'action': 'stop',
                    'entropy': current_entropy,
                    'reason': f"Entropy {current_entropy:.3f} > threshold {threshold}"
                })
                continue  # Don't explore further from this node
            
            # Add neighbors to queue
            neighbors = adjacency.get(current_node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
                    steps.append({
                        'step': len(steps),
                        'node': neighbor,
                        'action': 'queue',
                        'entropy': entropy_values.get(neighbor, 0.5),
                        'depth': depth + 1
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
        """Evaluate model predictions against ground truth"""
        
        if not predictions or not ground_truth:
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