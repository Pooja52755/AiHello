import json
import random
import numpy as np
from typing import List, Dict, Any
from faker import Faker
import networkx as nx

class TrainingDataGenerator:
    def __init__(self, num_graphs: int = 100, max_nodes: int = 50, max_sentences: int = 10):
        self.fake = Faker()
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.max_sentences = max_sentences
        self.vocab = self._generate_vocab(1000)  # Generate 1000 random words
    
    def _generate_vocab(self, size: int) -> List[str]:
        """Generate a vocabulary of random words"""
        return [self.fake.word() for _ in range(size)]
    
    def generate_sentence(self, min_length: int = 3, max_length: int = 15) -> str:
        """Generate a random sentence"""
        length = random.randint(min_length, max_length)
        return ' '.join(random.sample(self.vocab, length)).capitalize() + '.'
    
    def generate_graph(self, graph_id: int) -> Dict[str, Any]:
        """Generate a single graph with nodes and edges"""
        num_nodes = random.randint(5, self.max_nodes)
        num_sentences = random.randint(1, self.max_sentences)
        
        # Generate random sentences
        sentences = [self.generate_sentence() for _ in range(num_sentences)]
        
        # Create nodes with random properties
        nodes = []
        node_counter = 0
        
        for i in range(num_nodes):
            # Each node can appear in 1-3 sentences
            sentence_ids = random.sample(
                range(num_sentences), 
                k=min(random.randint(1, 3), num_sentences)
            )
            
            node = {
                'id': f"node_{graph_id}_{node_counter}",
                'label': random.choice(self.vocab),
                'sentence_ids': sentence_ids,
                'degree': random.randint(1, 10),
                'betweenness': random.random(),
                'is_boundary': random.random() > 0.8  # 20% chance of being a boundary
            }
            nodes.append(node)
            node_counter += 1
        
        # Create edges (random for now, but could be made more structured)
        edges = []
        for i in range(num_nodes * 2):  # Roughly 2 edges per node
            source = random.choice(nodes)['id']
            target = random.choice(nodes)['id']
            if source != target:  # No self-loops
                edges.append({
                    'source': source,
                    'target': target,
                    'label': random.choice(['is_a', 'has_a', 'related_to'])
                })
        
        return {
            'id': f"graph_{graph_id}",
            'nodes': nodes,
            'edges': edges,
            'sentences': sentences
        }
    
    def generate_dataset(self) -> List[Dict]:
        """Generate the complete dataset"""
        graphs = []
        for i in range(self.num_graphs):
            graphs.append(self.generate_graph(i))
        return graphs
    
    def save_dataset(self, filepath: str):
        """Save the generated dataset to a file"""
        dataset = self.generate_dataset()
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filepath} with {len(dataset)} graphs")

def main():
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate and save training data
    generator = TrainingDataGenerator(num_graphs=200, max_nodes=30, max_sentences=8)
    generator.save_dataset('data/training_graphs.json')
    
    # Also generate a smaller test set
    test_generator = TrainingDataGenerator(num_graphs=50, max_nodes=20, max_sentences=5)
    test_generator.save_dataset('data/test_graphs.json')
    
    print("\nTo train the model, run:")
    print("python train_entropy_model.py")

if __name__ == "__main__":
    main()
