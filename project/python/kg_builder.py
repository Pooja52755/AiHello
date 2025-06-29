"""
Enhanced Knowledge Graph Builder for Sentence Boundary Detection

This module builds knowledge graphs from SVO triplets with enhanced features
for sentence boundary detection using entropy-based traversal.
"""

import networkx as nx
import pickle
import json
from typing import List, Dict, Any, Set, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraphBuilder:
    """
    Enhanced knowledge graph builder with support for:
    - Rich node and edge attributes
    - Sentence boundary preservation
    - Graph serialization and loading
    - Multiple graph formats for different use cases
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.sentence_mappings = {}  # Maps nodes to sentence IDs
        self.node_attributes = {}
        self.edge_attributes = {}
        
    def build_graph(self, triplets: List[Dict], sentences: List[str]) -> Dict[str, Any]:
        """
        Build enhanced knowledge graph from SVO triplets
        
        Args:
            triplets: List of SVO triplet dictionaries
            sentences: List of original sentences
            
        Returns:
            Dictionary containing graph data and metadata
        """
        
        self.graph.clear()
        self.sentence_mappings.clear()
        self.node_attributes.clear()
        self.edge_attributes.clear()
        
        # Statistics tracking
        stats = {
            'total_triplets': len(triplets),
            'total_sentences': len(sentences),
            'unique_subjects': set(),
            'unique_objects': set(),
            'unique_verbs': set()
        }
        
        # Build graph from triplets
        for i, triplet in enumerate(triplets):
            subject = triplet['subject'].strip()
            verb = triplet['verb'].strip()
            obj = triplet['object'].strip()
            sentence_id = triplet.get('sentence_id', 0)
            confidence = triplet.get('confidence', 1.0)
            
            # Skip invalid triplets (but allow NONE objects)
            if not subject or not verb:
                continue
                
            stats['unique_subjects'].add(subject)
            if obj != "NONE":
                stats['unique_objects'].add(obj)
            stats['unique_verbs'].add(verb)
            
            # Add nodes with enhanced attributes
            self._add_enhanced_node(subject, sentence_id, 'subject')
            if obj != "NONE":
                self._add_enhanced_node(obj, sentence_id, 'object')
            
            # Add edge with enhanced attributes
            if obj != "NONE":
                self._add_enhanced_edge(subject, obj, verb, sentence_id, confidence)
            else:
                # For triplets with no object, create a virtual action node
                action_node = f"{subject}_{verb}_{sentence_id}"
                self._add_enhanced_node(action_node, sentence_id, 'action')
                self._add_enhanced_edge(subject, action_node, verb, sentence_id, confidence)
        
        # Calculate graph metrics
        self._calculate_graph_metrics()
        
        # Prepare output format
        graph_data = self._prepare_graph_data()
        
        # Update statistics
        stats.update({
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'unique_subjects': len(stats['unique_subjects']),
            'unique_objects': len(stats['unique_objects']),
            'unique_verbs': len(stats['unique_verbs']),
            'density': nx.density(self.graph) if len(self.graph.nodes()) > 0 else 0.0,
            'is_connected': nx.is_weakly_connected(self.graph) if len(self.graph.nodes()) > 1 else len(self.graph.nodes()) == 1
        })
        
        return {
            'graph': graph_data,
            'sentence_mappings': dict(self.sentence_mappings),
            'statistics': stats,
            'networkx_graph': self.graph.copy()  # For advanced analysis
        }
    
    def _add_enhanced_node(self, node_id: str, sentence_id: int, node_type: str):
        """Add node with enhanced attributes"""
        
        if node_id not in self.graph:
            self.graph.add_node(node_id)
            self.node_attributes[node_id] = {
                'id': node_id,
                'label': node_id,
                'type': node_type,
                'sentence_ids': set(),
                'degree': 0,
                'in_degree': 0,
                'out_degree': 0,
                'betweenness': 0.0,
                'closeness': 0.0,
                'pagerank': 0.0,
                'clustering': 0.0
            }
            self.sentence_mappings[node_id] = []
        
        # Update sentence mapping
        if sentence_id not in self.sentence_mappings[node_id]:
            self.sentence_mappings[node_id].append(sentence_id)
        
        self.node_attributes[node_id]['sentence_ids'].add(sentence_id)
    
    def _add_enhanced_edge(self, source: str, target: str, verb: str, 
                          sentence_id: int, confidence: float):
        """Add edge with enhanced attributes"""
        
        edge_key = f"{source}-{verb}->{target}"
        
        self.graph.add_edge(source, target, 
                           verb=verb, 
                           sentence_id=sentence_id,
                           confidence=confidence,
                           edge_id=edge_key)
        
        self.edge_attributes[edge_key] = {
            'source': source,
            'target': target,
            'verb': verb,
            'sentence_id': sentence_id,
            'confidence': confidence,
            'weight': confidence
        }
    
    def _calculate_graph_metrics(self):
        """Calculate enhanced graph metrics for all nodes"""
        
        if len(self.graph) == 0:
            return
        
        try:
            # Centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            
            # For large graphs, sample for expensive computations
            if len(self.graph) > 1000:
                sample_nodes = list(self.graph.nodes())[:1000]
                betweenness = nx.betweenness_centrality(self.graph, k=sample_nodes)
                closeness = nx.closeness_centrality(self.graph)
            else:
                betweenness = nx.betweenness_centrality(self.graph)
                closeness = nx.closeness_centrality(self.graph)
            
            pagerank = nx.pagerank(self.graph, max_iter=100)
            clustering = nx.clustering(self.graph.to_undirected())
            
            # Update node attributes
            for node in self.graph.nodes():
                if node in self.node_attributes:
                    self.node_attributes[node].update({
                        'degree': self.graph.degree(node),
                        'in_degree': self.graph.in_degree(node),
                        'out_degree': self.graph.out_degree(node),
                        'degree_centrality': degree_centrality.get(node, 0.0),
                        'betweenness': betweenness.get(node, 0.0),
                        'closeness': closeness.get(node, 0.0),
                        'pagerank': pagerank.get(node, 0.0),
                        'clustering': clustering.get(node, 0.0)
                    })
                    
        except Exception as e:
            logger.warning(f"Error calculating graph metrics: {e}")
    
    def _prepare_graph_data(self) -> Dict[str, Any]:
        """Prepare graph data for visualization and analysis"""
        
        nodes = []
        edges = []
        
        # Prepare nodes
        for node_id in self.graph.nodes():
            node_data = self.node_attributes.get(node_id, {})
            
            # Convert sets to lists for JSON serialization
            if 'sentence_ids' in node_data:
                node_data['sentence_ids'] = list(node_data['sentence_ids'])
            
            nodes.append(node_data)
        
        # Prepare edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge_info = {
                'source': source,
                'target': target,
                'verb': edge_data.get('verb', ''),
                'sentence_id': edge_data.get('sentence_id', 0),
                'confidence': edge_data.get('confidence', 1.0),
                'weight': edge_data.get('confidence', 1.0)
            }
            edges.append(edge_info)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'directed': True
        }
    
    def get_sentence_subgraph(self, sentence_id: int) -> nx.Graph:
        """Extract subgraph for a specific sentence"""
        
        sentence_nodes = [node for node, attrs in self.node_attributes.items()
                         if sentence_id in attrs.get('sentence_ids', set())]
        
        return self.graph.subgraph(sentence_nodes).copy()
    
    def get_node_sentence_mapping(self) -> Dict[str, List[int]]:
        """Get mapping of nodes to their sentence IDs"""
        return dict(self.sentence_mappings)
    
    def save_graph(self, filepath: str, format: str = 'pickle') -> None:
        """
        Save graph in various formats
        
        Args:
            filepath: Path to save file
            format: 'pickle', 'json', 'gml', 'graphml'
        """
        
        filepath = Path(filepath)
        
        if format == 'pickle':
            graph_data = {
                'graph': self.graph,
                'sentence_mappings': self.sentence_mappings,
                'node_attributes': self.node_attributes,
                'edge_attributes': self.edge_attributes
            }
            with open(filepath, 'wb') as f:
                pickle.dump(graph_data, f)
                
        elif format == 'json':
            graph_data = self._prepare_graph_data()
            graph_data['sentence_mappings'] = dict(self.sentence_mappings)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
                
        elif format == 'gml':
            nx.write_gml(self.graph, filepath)
            
        elif format == 'graphml':
            nx.write_graphml(self.graph, filepath)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Graph saved to {filepath} in {format} format")
    
    def load_graph(self, filepath: str, format: str = 'pickle') -> None:
        """Load graph from file"""
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                graph_data = pickle.load(f)
            
            self.graph = graph_data['graph']
            self.sentence_mappings = graph_data['sentence_mappings']
            self.node_attributes = graph_data['node_attributes']
            self.edge_attributes = graph_data['edge_attributes']
            
        elif format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            self._load_from_json_data(graph_data)
            
        elif format == 'gml':
            self.graph = nx.read_gml(filepath)
            self._rebuild_attributes()
            
        elif format == 'graphml':
            self.graph = nx.read_graphml(filepath)
            self._rebuild_attributes()
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Graph loaded from {filepath}")
    
    def _load_from_json_data(self, graph_data: Dict[str, Any]):
        """Load graph from JSON data"""
        
        self.graph.clear()
        
        # Load nodes
        for node_data in graph_data.get('nodes', []):
            node_id = node_data['id']
            self.graph.add_node(node_id)
            self.node_attributes[node_id] = node_data
            
            # Convert sentence_ids back to set
            if 'sentence_ids' in node_data:
                node_data['sentence_ids'] = set(node_data['sentence_ids'])
        
        # Load edges
        for edge_data in graph_data.get('edges', []):
            source = edge_data['source']
            target = edge_data['target']
            
            self.graph.add_edge(source, target, **{
                k: v for k, v in edge_data.items() 
                if k not in ['source', 'target']
            })
        
        # Load sentence mappings
        self.sentence_mappings = graph_data.get('sentence_mappings', {})
    
    def _rebuild_attributes(self):
        """Rebuild attributes after loading from basic formats"""
        
        self.node_attributes = {}
        self.sentence_mappings = {}
        
        for node in self.graph.nodes():
            self.node_attributes[node] = {
                'id': node,
                'label': node,
                'type': 'entity',
                'sentence_ids': set(),
                'degree': self.graph.degree(node)
            }
        
        self._calculate_graph_metrics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        if len(self.graph) == 0:
            return {'empty_graph': True}
        
        # Basic stats
        stats = {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph)
        }
        
        # Degree statistics
        degrees = [d for n, d in self.graph.degree()]
        if degrees:
            stats.update({
                'avg_degree': sum(degrees) / len(degrees),
                'max_degree': max(degrees),
                'min_degree': min(degrees)
            })
        
        # Sentence distribution
        sentence_counts = {}
        for node, sentence_ids in self.sentence_mappings.items():
            for sid in sentence_ids:
                sentence_counts[sid] = sentence_counts.get(sid, 0) + 1
        
        if sentence_counts:
            stats.update({
                'num_sentences': len(sentence_counts),
                'avg_nodes_per_sentence': sum(sentence_counts.values()) / len(sentence_counts),
                'max_nodes_in_sentence': max(sentence_counts.values()),
                'min_nodes_in_sentence': min(sentence_counts.values())
            })
        
        return stats

# Maintain backward compatibility
class KnowledgeGraphBuilder(EnhancedKnowledgeGraphBuilder):
    """Backward compatible wrapper"""
    pass