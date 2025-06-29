import json
import torch
import numpy as np
from typing import Dict, List, Any
from train_entropy_model import EntropyModelTrainer, BoundaryDetectionDataset
from torch.utils.data import DataLoader

class EntropyInference:
    def __init__(self, model_path: str = 'models/boundary_predictor.pth'):
        """Initialize the inference with a trained model"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.trainer = EntropyModelTrainer.load_model(model_path)
        self.model = self.trainer.model
        self.model.eval()
        
    def predict_boundaries(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict boundary nodes for a given graph"""
        # Prepare dataset (single graph)
        dataset = BoundaryDetectionDataset([graph_data])
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Get predictions
        all_preds = []
        all_nodes = []
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch['x'].to(self.device)
                outputs = self.model(x)
                all_preds.extend(outputs.cpu().numpy())
                all_nodes.extend(batch['node_id'])
        
        # Create node_id to prediction mapping
        node_predictions = {
            node_id: float(pred) 
            for node_id, pred in zip(all_nodes, all_preds)
        }
        
        # Update graph data with predictions
        for node in graph_data['nodes']:
            node_id = node['id']
            node['boundary_score'] = node_predictions.get(node_id, 0.0)
            node['is_boundary'] = node['boundary_score'] > 0.5
        
        return graph_data
    
    def find_sentence_boundaries(self, graph_data: Dict[str, Any], 
                               starting_node: str, 
                               threshold: float = 0.5) -> Dict[str, Any]:
        """
        Find sentence boundaries starting from a given node
        
        Args:
            graph_data: The input graph data
            starting_node: ID of the starting node
            threshold: Threshold for boundary prediction (0-1)
            
        Returns:
            Dict containing traversal path and boundary nodes
        """
        # First, get boundary predictions for all nodes
        graph_data = self.predict_boundaries(graph_data)
        
        # Create a mapping of node IDs to their data
        node_map = {node['id']: node for node in graph_data['nodes']}
        
        # Create a graph for traversal
        G = nx.Graph()
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'])
        
        # Initialize BFS
        visited = set()
        boundary_nodes = []
        traversal_path = []
        queue = [(starting_node, 0)]  # (node_id, depth)
        
        while queue:
            current_node, depth = queue.pop(0)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            node_data = node_map.get(current_node, {})
            
            # Add to traversal path
            traversal_path.append({
                'node': current_node,
                'depth': depth,
                'boundary_score': node_data.get('boundary_score', 0.0),
                'is_boundary': node_data.get('is_boundary', False)
            })
            
            # Check if this is a boundary node
            if node_data.get('is_boundary', False) and depth > 0:
                boundary_nodes.append(current_node)
                continue  # Don't explore beyond boundary nodes
            
            # Add neighbors to queue
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited and neighbor not in [n for n, _ in queue]:
                    queue.append((neighbor, depth + 1))
        
        return {
            'traversal_path': traversal_path,
            'boundary_nodes': boundary_nodes,
            'visited_nodes': list(visited),
            'threshold_used': threshold
        }

def load_graph_data(filepath: str) -> Dict[str, Any]:
    """Load graph data from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run boundary detection inference')
    parser.add_argument('--model', type=str, default='models/boundary_predictor.pth',
                       help='Path to trained model')
    parser.add_argument('--graph', type=str, required=True,
                       help='Path to graph data JSON file')
    parser.add_argument('--start-node', type=str, required=True,
                       help='Starting node ID')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Boundary detection threshold (0-1)')
    
    args = parser.parse_args()
    
    try:
        # Load graph data
        print(f"Loading graph data from {args.graph}...")
        graph_data = load_graph_data(args.graph)
        
        # Initialize inference
        print(f"Loading model from {args.model}...")
        inferencer = EntropyInference(args.model)
        
        # Find sentence boundaries
        print(f"Finding boundaries starting from node {args.start_node}...")
        result = inferencer.find_sentence_boundaries(
            graph_data,
            args.start_node,
            args.threshold
        )
        
        # Print results
        print("\n=== Traversal Results ===")
        print(f"Starting node: {args.start_node}")
        print(f"Visited {len(result['visited_nodes'])} nodes")
        print(f"Found {len(result['boundary_nodes'])} boundary nodes")
        print("\nBoundary nodes:")
        for node_id in result['boundary_nodes']:
            print(f"- {node_id}")
        
        # Save results
        output_file = 'boundary_detection_results.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
