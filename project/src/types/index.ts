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

export interface TraversalResult {
  traversal_path: string[];
  visited_nodes: string[];
  boundary_nodes: string[];
  entropy_scores: { [nodeId: string]: number };
  boundary_predictions: { [nodeId: string]: boolean };
  metrics: {
    total_nodes: number;
    visited_count: number;
    boundary_count: number;
    efficiency: number;
    entropy_threshold: number;
  };
}

export interface EvaluationMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  accuracy: number;
}

export interface TraversalApiResponse {
  traversal_result: TraversalResult;
  evaluation_metrics: EvaluationMetrics;
  start_node: string;
  entropy_threshold: number;
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

// Legacy support
export interface Node {
  id: string;
  label: string;
  type?: string;
}

export interface Edge {
  source: string;
  target: string;
  label?: string;
}