import requests
import json

def test_process_text():
    """Test the /api/process-text endpoint"""
    url = "http://localhost:5000/api/process-text"
    
    # Sample text from War and Peace
    sample_text = """
    "Well, Prince, so Genoa and Lucca are now just family estates of the 
    Buonapartes. But I warn you, if you don't tell me that this means war, 
    if you still try to defend the infamies and horrors perpetrated by that 
    Antichrist--I really believe he is Antichrist--I will have nothing more 
    to do with you and you are no longer my friend, no longer my 'faithful 
    slave,' as you call yourself!"
    """
    
    data = {"text": sample_text}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        # Save the result
        with open('api_test_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        print("API test successful! Results saved to 'api_test_result.json'")
        print(f"Processed {len(result['sentences'])} sentences")
        print(f"Extracted {len(result['triplets'])} SVO triplets")
        print(f"Graph contains {len(result['graph']['nodes'])} nodes and {len(result['graph']['edges'])} edges")
        
        return result
    except Exception as e:
        print(f"Error testing API: {e}")
        return None

def test_traverse_graph(graph_data, starting_nodes):
    """Test the /api/traverse-graph endpoint"""
    url = "http://localhost:5000/api/traverse-graph"
    
    data = {
        "graph": graph_data,
        "starting_nodes": starting_nodes,
        "entropy_threshold": 0.7
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        # Save the result
        with open('traversal_test_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        print("\nTraversal test successful! Results saved to 'traversal_test_result.json'")
        print(f"Traversed from {len(starting_nodes)} starting nodes")
        
        return result
    except Exception as e:
        print(f"Error testing traversal: {e}")
        return None

if __name__ == "__main__":
    # First test processing text
    print("Testing text processing...")
    result = test_process_text()
    
    if result and 'graph' in result and 'nodes' in result['graph'] and result['graph']['nodes']:
        # If we have a valid graph, test traversal
        print("\nTesting graph traversal...")
        # Get the first 3 nodes as starting points
        starting_nodes = [node['id'] for node in result['graph']['nodes'][:3]]
        test_traverse_graph(result['graph'], starting_nodes)
