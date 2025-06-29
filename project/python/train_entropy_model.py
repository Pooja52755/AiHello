import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PyGData
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional
import networkx as nx
from tqdm import tqdm

class BoundaryDetectionDataset(Dataset):
    """Dataset for training the boundary detection model"""
    def __init__(self, graph_data: List[Dict], window_size: int = 3):
        self.graph_data = graph_data
        self.window_size = window_size
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self) -> List[Dict]:
        """Convert graph data into training samples"""
        samples = []
        
        for graph in tqdm(self.graph_data, desc="Preparing samples"):
            # Extract node features and edges
            nodes = {node['id']: node for node in graph['nodes']}
            edges = [(e['source'], e['target']) for e in graph['edges']]
            
            # Create NetworkX graph for easier traversal
            G = nx.Graph()
            G.add_edges_from(edges)
            
            # For each node, create positive and negative samples
            for node_id, node_data in nodes.items():
                # Get node features
                features = self._get_node_features(node_id, node_data, G)
                
                # Get boundary label (1 if node is a boundary, else 0)
                is_boundary = 1 if node_data.get('is_boundary', False) else 0
                
                # Add to samples
                samples.append({
                    'features': features,
                    'label': is_boundary,
                    'node_id': node_id,
                    'graph_id': graph.get('id', 'unknown')
                })
                
        return samples
    
    def _get_node_features(self, node_id: str, node_data: Dict, G: nx.Graph) -> torch.Tensor:
        """Extract features for a node"""
        features = []
        
        # 1. Node degree features
        degree = G.degree(node_id)
        features.append(degree)
        
        # 2. Betweenness centrality (if available)
        betweenness = node_data.get('betweenness', 0)
        features.append(betweenness)
        
        # 3. Sentence count (how many sentences this node appears in)
        sentence_count = len(node_data.get('sentence_ids', []))
        features.append(sentence_count)
        
        # 4. Neighbor degrees
        neighbor_degrees = [G.degree(n) for n in G.neighbors(node_id)]
        features.append(np.mean(neighbor_degrees) if neighbor_degrees else 0)
        features.append(np.std(neighbor_degrees) if neighbor_degrees else 0)
        
        # 5. Local clustering coefficient
        clustering = nx.clustering(G, node_id)
        features.append(clustering)
        
        return torch.FloatTensor(features)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'x': sample['features'],
            'y': torch.tensor(sample['label'], dtype=torch.float),
            'node_id': sample['node_id'],
            'graph_id': sample['graph_id']
        }

class BoundaryPredictor(nn.Module):
    """Neural network for boundary prediction"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super(BoundaryPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mlp(x).squeeze()

class EntropyModelTrainer:
    """Class for training the entropy-based boundary detection model"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'hidden_dim': 64,
            'dropout': 0.3,
            'weight_decay': 1e-5,
            'test_size': 0.2,
            'random_state': 42,
            'model_save_path': 'models/boundary_predictor.pth',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
    
    def train(self, graph_data: List[Dict]):
        """Train the boundary prediction model"""
        # Prepare dataset
        dataset = BoundaryDetectionDataset(graph_data)
        
        # Split into train and validation sets
        train_data, val_data = train_test_split(
            dataset.samples,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        train_dataset = torch.utils.data.Subset(dataset, range(len(train_data)))
        val_dataset = torch.utils.data.Subset(dataset, range(len(train_data), len(train_data) + len(val_data)))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # Initialize model
        input_dim = len(dataset[0]['x'])
        self.model = BoundaryPredictor(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            dropout=self.config['dropout']
        ).to(self.config['device'])
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                x = batch['x'].to(self.config['device'])
                y = batch['y'].to(self.config['device'])
                
                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * x.size(0)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            train_loss = train_loss / len(train_loader.dataset)
            
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics: {val_metrics}")
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                print(f"  Model saved to {self.config['model_save_path']}")
    
    def evaluate(self, data_loader):
        """Evaluate the model on the given data loader"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch['x'].to(self.config['device'])
                y = batch['y'].to(self.config['device'])
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                val_loss += loss.item() * x.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        labels_binary = np.array(all_labels).astype(int)
        
        metrics = {
            'loss': val_loss / len(data_loader.dataset),
            'accuracy': accuracy_score(labels_binary, preds_binary),
            'precision': precision_score(labels_binary, preds_binary, zero_division=0),
            'recall': recall_score(labels_binary, preds_binary, zero_division=0),
            'f1': f1_score(labels_binary, preds_binary, zero_division=0)
        }
        
        return metrics['loss'], metrics
    
    def save_model(self):
        """Save the model and tokenizer"""
        if not self.model:
            raise ValueError("Model not initialized. Call train() first.")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, self.config['model_save_path'])
    
    @classmethod
    def load_model(cls, model_path: str):
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        
        # Create trainer with loaded config
        trainer = cls(config)
        
        # Initialize model
        input_dim = config.get('input_dim', 10)  # Default input dim if not found
        trainer.model = BoundaryPredictor(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        ).to(config['device'])
        
        # Load weights
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        return trainer

def prepare_training_data(graph_data_path: str) -> List[Dict]:
    """Prepare training data from graph data"""
    with open(graph_data_path, 'r') as f:
        graph_data = json.load(f)
    
    # Add boundary labels if not present
    for graph in graph_data:
        # This is a simplified example - in practice, you'd want to use
        # ground truth boundaries from your dataset
        for node in graph['nodes']:
            # Randomly assign some nodes as boundaries for demonstration
            node['is_boundary'] = random.random() > 0.8
    
    return graph_data

def main():
    # Example usage
    graph_data_path = "data/training_graphs.json"  # Update this path
    
    try:
        # Prepare training data
        print("Preparing training data...")
        graph_data = prepare_training_data(graph_data_path)
        
        # Initialize and train model
        print("Initializing trainer...")
        trainer = EntropyModelTrainer()
        
        print("Starting training...")
        trainer.train(graph_data)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
