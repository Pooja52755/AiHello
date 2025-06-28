import networkx as nx
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_graph(self, triplets: List[Dict], sentences: List[str]) -> Dict[str, Any]:
        """Build knowledge graph from SVO triplets"""
        self.graph.clear()
        
        # Node and edge data for visualization
        nodes = {}
        edges = []
        sentence_mapping = {}
        
        for triplet in triplets:
            subject = triplet['subject']
            verb = triplet['verb']
            obj = triplet['object']
            sentence_id = triplet.get('sentence_id', 0)
            confidence = triplet.get('confidence', 0.5)
            
            # Add nodes
            if subject not in nodes:
                nodes[subject] = {
                    'id': subject,
                    'label': subject,
                    'type': 'entity',
                    'sentence_ids': set(),
                    'degree': 0,
                    'betweenness': 0
                }
            nodes[subject]['sentence_ids'].add(sentence_id)
            
            if obj != "NONE" and obj not in nodes:
                nodes[obj] = {
                    'id': obj,
                    'label': obj,
                    'type': 'entity',
                    'sentence_ids': set(),
                    'degree': 0,
                    'betweenness': 0
                }
                nodes[obj]['sentence_ids'].add(sentence_id)
            
            # Add edge to NetworkX graph
            if obj != "NONE":
                self.graph.add_edge(subject, obj, 
                                  verb=verb, 
                                  confidence=confidence,
                                  sentence_id=sentence_id)
                
                # Add edge for visualization
                edges.append({
                    'source': subject,
                    'target': obj,
                    'label': verb,
                    'confidence': confidence,
                    'sentence_id': sentence_id
                })
        
        # Calculate graph metrics
        self._calculate_graph_metrics(nodes)
        
        # Convert sentence_ids sets to lists for JSON serialization
        for node in nodes.values():
            node['sentence_ids'] = list(node['sentence_ids'])
        
        # Create sentence mapping
        for i, sentence in enumerate(sentences):
            sentence_mapping[i] = {
                'id': i,
                'text': sentence,
                'nodes': [node_id for node_id, node_data in nodes.items() 
                         if i in node_data['sentence_ids']]
            }
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'sentences': sentence_mapping,
            'stats': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
                'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
            }
        }
    
    def _calculate_graph_metrics(self, nodes: Dict):
        """Calculate graph centrality metrics"""
        if self.graph.number_of_nodes() == 0:
            return
        
        try:
            # Calculate degree centrality
            degree_centrality = nx.degree_centrality(self.graph)
            
            # Calculate betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # Update node data
            for node_id in nodes:
                if node_id in degree_centrality:
                    nodes[node_id]['degree'] = degree_centrality[node_id]
                if node_id in betweenness_centrality:
                    nodes[node_id]['betweenness'] = betweenness_centrality[node_id]
                    
        except Exception as e:
            logger.warning(f"Error calculating graph metrics: {e}")
    
    def get_node_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node"""
        if node_id not in self.graph:
            return []
        
        # Get both predecessors and successors
        neighbors = list(self.graph.predecessors(node_id)) + list(self.graph.successors(node_id))
        return list(set(neighbors))  # Remove duplicates
    
    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """Get shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph.to_undirected(), source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_subgraph(self, nodes: List[str]) -> nx.DiGraph:
        """Get subgraph containing specified nodes"""
        return self.graph.subgraph(nodes)