from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from kg_builder import KnowledgeGraphBuilder
from entropy_detector import EntropyDetector
from text_processor import TextProcessor
from node_embeddings import embedding_generator
import json
import os

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
text_processor = TextProcessor()
kg_builder = KnowledgeGraphBuilder()
entropy_detector = EntropyDetector()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "KG Entropy Detection API is running"})

@app.route('/api/process-text', methods=['POST'])
def process_text():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Process text and build knowledge graph
        logger.info(f"Processing text of length: {len(text)}")
        
        # Extract sentences and SVO triplets
        sentences = text_processor.extract_sentences(text)
        triplets = text_processor.extract_svo_triplets(text)
        
        # Build knowledge graph
        kg_data = kg_builder.build_graph(triplets, sentences)
        
        # Calculate initial entropy values
        entropy_data = entropy_detector.calculate_initial_entropy(kg_data)
        
        # Create JSON-serializable graph data (exclude NetworkX object)
        graph_for_api = {
            'nodes': kg_data['graph']['nodes'],
            'edges': kg_data['graph']['edges']
        }
        
        response = {
            "sentences": sentences,
            "triplets": triplets,
            "graph": graph_for_api,
            "entropy": entropy_data,
            "stats": {
                "num_sentences": len(sentences),
                "num_triplets": len(triplets),
                "num_nodes": len(kg_data['graph']['nodes']),
                "num_edges": len(kg_data['graph']['edges'])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/traverse-graph', methods=['POST'])
def traverse_graph():
    try:
        data = request.json
        graph_data = data.get('graph', {})
        starting_nodes = data.get('starting_nodes', [])
        entropy_threshold = data.get('entropy_threshold', 0.4)
        
        if not graph_data or not starting_nodes:
            return jsonify({"error": "Graph data and starting nodes required"}), 400
        
        # Use the first starting node for traversal
        start_node = starting_nodes[0]
        
        # Perform entropy-guided traversal
        traversal_result = entropy_detector.entropy_guided_traversal(
            graph_data, start_node, entropy_threshold
        )
        
        # Calculate evaluation metrics
        boundary_nodes = set(traversal_result['boundary_nodes'])
        evaluation_metrics = entropy_detector.calculate_evaluation_metrics(boundary_nodes)
        
        response = {
            "traversal_result": traversal_result,
            "evaluation_metrics": evaluation_metrics,
            "start_node": start_node,
            "entropy_threshold": entropy_threshold
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in graph traversal: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/calculate-entropy', methods=['POST'])
def calculate_entropy():
    """Calculate entropy scores for all nodes"""
    try:
        data = request.json
        graph_data = data.get('graph', {})
        start_node = data.get('start_node', None)
        
        if not graph_data:
            return jsonify({"error": "Graph data required"}), 400
        
        # Calculate entropy scores
        entropy_scores = entropy_detector.calculate_entropy_scores(graph_data, start_node)
        
        # Calculate neighborhood entropies
        neighborhood_entropies = {}
        for node in graph_data['nodes']:
            node_id = node['id']
            neighborhood_entropies[node_id] = entropy_detector.calculate_neighborhood_entropy(graph_data, node_id)
        
        response = {
            "entropy_scores": entropy_scores,
            "neighborhood_entropies": neighborhood_entropies,
            "node_embeddings_count": len(entropy_detector.node_embeddings),
            "start_node": start_node
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error calculating entropy: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
        # Perform entropy-based traversal
        traversal_results = entropy_detector.traverse_with_entropy(
            graph_data, starting_nodes, entropy_threshold
        )
        
        return jsonify(traversal_results)
        
    except Exception as e:
        logger.error(f"Error traversing graph: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate-model', methods=['POST'])
def evaluate_model():
    try:
        data = request.json
        predictions = data.get('predictions', [])
        ground_truth = data.get('ground_truth', [])
        
        # Calculate evaluation metrics
        metrics = entropy_detector.evaluate_predictions(predictions, ground_truth)
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/load-gutenberg', methods=['POST'])
def load_gutenberg():
    try:
        data = request.json
        book_id = data.get('book_id', '2600')  # Default: War and Peace
        
        # Load text from Project Gutenberg
        text = text_processor.load_gutenberg_text(book_id)
        
        return jsonify({"text": text, "book_id": book_id})
        
    except Exception as e:
        logger.error(f"Error loading Gutenberg text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/embeddings', methods=['POST'])
def generate_embeddings():
    """Generate and return node embeddings"""
    try:
        data = request.json
        graph_data = data.get('graph', {})
        
        if not graph_data:
            return jsonify({"error": "Graph data required"}), 400
        
        # Generate embeddings
        embeddings = embedding_generator.generate_node_embeddings(graph_data)
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to file
        embeddings_file = os.path.join(data_dir, 'node_embeddings.json')
        embedding_generator.save_embeddings(embeddings, embeddings_file)
        
        response = {
            "embeddings": embeddings,
            "count": len(embeddings),
            "embedding_dimension": len(list(embeddings.values())[0]) if embeddings else 0,
            "saved_to": embeddings_file
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/entropy-score', methods=['POST'])
def calculate_entropy_between_nodes():
    """Calculate entropy score between two specific nodes"""
    try:
        data = request.json
        node1_embedding = data.get('node1_embedding', [])
        node2_embedding = data.get('node2_embedding', [])
        
        if not node1_embedding or not node2_embedding:
            return jsonify({"error": "Both node embeddings required"}), 400
        
        # Calculate entropy score
        entropy_score = embedding_generator.calculate_entropy_score(node1_embedding, node2_embedding)
        similarity = embedding_generator.calculate_cosine_similarity(node1_embedding, node2_embedding)
        
        response = {
            "entropy_score": entropy_score,
            "cosine_similarity": similarity,
            "calculation": "entropy = 1 - cosine_similarity"
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error calculating entropy score: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)