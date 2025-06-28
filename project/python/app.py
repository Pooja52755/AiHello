from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from kg_builder import KnowledgeGraphBuilder
from entropy_detector import EntropyDetector
from text_processor import TextProcessor
import json

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
        
        response = {
            "sentences": sentences,
            "triplets": triplets,
            "graph": kg_data,
            "entropy": entropy_data,
            "stats": {
                "num_sentences": len(sentences),
                "num_triplets": len(triplets),
                "num_nodes": len(kg_data.get('nodes', [])),
                "num_edges": len(kg_data.get('edges', []))
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
        entropy_threshold = data.get('entropy_threshold', 0.7)
        
        if not graph_data or not starting_nodes:
            return jsonify({"error": "Graph data and starting nodes required"}), 400
        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)