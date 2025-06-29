import json
import sys
from text_processor import TextProcessor
from kg_builder import KnowledgeGraphBuilder
from entropy_detector import EntropyDetector

def test_enhanced_entropy():
    # Sample text for testing
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog barks loudly.
    A cat watches from the window. The fox and the dog are now friends.
    They play together in the garden behind the house.
    """
    
    print("Processing sample text...")
    
    # Initialize components
    print("Initializing components...")
    text_processor = TextProcessor()
    kg_builder = KnowledgeGraphBuilder()
    entropy_detector = EntropyDetector()
    
    # Process text
    print("Extracting sentences and triplets...")
    sentences = text_processor.extract_sentences(sample_text)
    triplets = []
    for sent in sentences:
        triplets.extend(text_processor.extract_svo_triplets(sent))
    
    print(f"Extracted {len(sentences)} sentences and {len(triplets)} triplets")
    
    # Build knowledge graph
    print("Building knowledge graph...")
    graph_data = kg_builder.build_graph(triplets, sentences)
    
    # Get starting nodes (first subject in first sentence)
    starting_nodes = []
    if triplets:
        starting_nodes.append(triplets[0]['subject'])
    
    if not starting_nodes:
        print("No starting nodes found!")
        return
    
    print(f"Starting traversal from nodes: {starting_nodes}")
    
    # Perform entropy-based traversal
    print("\nPerforming entropy-based traversal...")
    results = entropy_detector.traverse_with_entropy(
        graph_data=graph_data,
        starting_nodes=starting_nodes,
        threshold=0.6
    )
    
    # Print results
    print("\n=== Traversal Results ===")
    for i, result in enumerate(results['traversal_results']):
        print(f"\nTraversal {i+1} from '{result['starting_node']}':")
        print(f"- Visited {len(result['visited_nodes'])} nodes")
        print(f"- Boundary nodes: {result['boundary_nodes']}")
        print(f"- Stopping reason: {result['stopping_reason']}")
        
        print("\nNode entropies:")
        for node in result['entropy_path']:
            boundary_marker = " [BOUNDARY]" if node.get('is_boundary', False) else ""
            print(f"  {node['node']}: {node.get('entropy', 0):.3f}{boundary_marker}")
    
    # Save results for visualization
    with open('traversal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to traversal_results.json")

if __name__ == "__main__":
    test_enhanced_entropy()
