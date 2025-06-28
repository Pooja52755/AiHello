import requests
from bs4 import BeautifulSoup
import json
from text_processor import TextProcessor
from kg_builder import KnowledgeGraphBuilder
from entropy_detector import EntropyDetector

def download_war_and_peace():
    """Download War and Peace from Project Gutenberg"""
    url = "https://www.gutenberg.org/files/2600/2600-0.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error downloading text: {e}")
        return None

def preprocess_text(text):
    """Clean and preprocess the text"""
    # Remove Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***"
    
    start_idx = text.find(start_marker) + len(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        return text
        
    text = text[start_idx:end_idx].strip()
    
    # Remove chapter headings and other metadata
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip() and 
                    not line.strip().startswith(('CHAPTER', 'Chapter', 'BOOK', 'Book'))]
    
    return '\n'.join(cleaned_lines)

def main():
    print("Downloading War and Peace from Project Gutenberg...")
    text = download_war_and_peace()
    
    if not text:
        print("Failed to download the text.")
        return
    
    print("Preprocessing text...")
    text = preprocess_text(text)
    
    # Process the first 5000 characters for demo purposes
    # Remove the slice to process the entire text (will take longer)
    sample_text = text[:5000]
    
    print("Processing text and building knowledge graph...")
    
    # Initialize components
    text_processor = TextProcessor()
    kg_builder = KnowledgeGraphBuilder()
    entropy_detector = EntropyDetector()
    
    # Extract sentences and build knowledge graph
    sentences = text_processor.extract_sentences(sample_text)
    triplets = []
    
    for i, sentence in enumerate(sentences):
        sentence_triplets = text_processor.extract_svo_triplets(sentence)
        for triplet in sentence_triplets:
            triplet['sentence_id'] = i
        triplets.extend(sentence_triplets)
    
    # Build knowledge graph
    kg_data = kg_builder.build_graph(triplets, sentences)
    
    # Save the knowledge graph
    with open('war_and_peace_kg.json', 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, indent=2)
    
    # Example: Get starting nodes (first 5 nodes)
    starting_nodes = [node['id'] for node in kg_data['nodes'][:5]]
    
    # Perform entropy-based traversal
    print("Performing entropy-based traversal...")
    traversal_results = entropy_detector.traverse_with_entropy(
        kg_data, 
        starting_nodes
    )
    
    # Save the results
    with open('traversal_results.json', 'w', encoding='utf-8') as f:
        json.dump(traversal_results, f, indent=2)
    
    print("\nProcessing complete!")
    print(f"- Processed {len(sentences)} sentences")
    print(f"- Extracted {len(triplets)} SVO triplets")
    print(f"- Built knowledge graph with {len(kg_data['nodes'])} nodes and {len(kg_data['edges'])} edges")
    print(f"- Results saved to 'war_and_peace_kg.json' and 'traversal_results.json'")

if __name__ == "__main__":
    main()
