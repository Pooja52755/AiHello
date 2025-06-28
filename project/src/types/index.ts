export interface GraphNode {
  id: string;
  label: string;
  type: string;
  sentence_ids: number[];
  degree: number;
  betweenness: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  confidence: number;
  sentence_id: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  sentences: { [key: number]: { id: number; text: string; nodes: string[] } };
  stats: {
    num_nodes: number;
    num_edges: number;
    density: number;
    is_connected: boolean;
  };
}

export interface TraversalStep {
  step: number;
  node: string;
  action: string;
  entropy: number;
  depth?: number;
  reason?: string;
}

export interface EntropyPath {
  node: string;
  entropy: number;
  depth: number;
}

export interface TraversalResultItem {
  starting_node: string;
  visited_nodes: string[];
  boundary_nodes: string[];
  entropy_path: EntropyPath[];
  traversal_steps: TraversalStep[];
  stopping_reason: string;
}

export interface TraversalResult {
  traversal_results: TraversalResultItem[];
  entropy_values: { [key: string]: number };
  threshold_used: number;
  summary: {
    total_traversals: number;
    avg_nodes_visited: number;
    avg_boundary_nodes: number;
  };
}

export interface ApiResponse {
  sentences: string[];
  triplets: Array<{
    subject: string;
    verb: string;
    object: string;
    sentence: string;
    sentence_id: number;
    confidence: number;
  }>;
  graph: GraphData;
  entropy: { [key: string]: number };
  stats: {
    num_sentences: number;
    num_triplets: number;
    num_nodes: number;
    num_edges: number;
  };
}