/**
 * Entropy computation utilities for knowledge graph nodes
 */

import { GraphData, Node, Edge } from '../types';

export interface EntropyScores {
  [nodeId: string]: number;
}

export interface EntropyResult {
  entropy_scores: EntropyScores;
  neighborhood_entropies: EntropyScores;
  node_embeddings_count: number;
  start_node: string | null;
}

/**
 * Calculate entropy scores for all nodes via API
 */
export async function calculateEntropyScores(
  graphData: GraphData,
  startNode?: string
): Promise<EntropyResult> {
  const response = await fetch('http://localhost:5001/api/calculate-entropy', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      graph: graphData,
      start_node: startNode,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to calculate entropy scores');
  }

  return response.json();
}

/**
 * Get color for node based on entropy score
 */
export function getEntropyColor(entropy: number): string {
  // Low entropy (0-0.3): Blue (same sentence)
  // Medium entropy (0.3-0.6): Yellow (transitional)
  // High entropy (0.6-1.0): Red (boundary)
  
  if (entropy <= 0.3) {
    return '#3B82F6'; // Blue
  } else if (entropy <= 0.6) {
    return '#F59E0B'; // Yellow
  } else {
    return '#EF4444'; // Red
  }
}

/**
 * Get opacity based on entropy score for visual emphasis
 */
export function getEntropyOpacity(entropy: number): number {
  // Higher entropy = more opaque
  return 0.4 + (entropy * 0.6);
}

/**
 * Classify node type based on entropy threshold
 */
export function classifyNodeType(
  nodeId: string,
  entropyScore: number,
  threshold: number,
  isVisited: boolean,
  isBoundary: boolean,
  isSelected: boolean
): 'default' | 'selected' | 'visited' | 'boundary' | 'low-entropy' | 'high-entropy' {
  if (isSelected) return 'selected';
  if (isBoundary) return 'boundary';
  if (isVisited) return 'visited';
  
  if (entropyScore > threshold) {
    return 'high-entropy';
  } else if (entropyScore < threshold * 0.5) {
    return 'low-entropy';
  }
  
  return 'default';
}

/**
 * Calculate entropy statistics for the graph
 */
export function calculateEntropyStats(entropyScores: EntropyScores): {
  mean: number;
  std: number;
  min: number;
  max: number;
  distribution: { low: number; medium: number; high: number };
} {
  const scores = Object.values(entropyScores);
  
  if (scores.length === 0) {
    return {
      mean: 0,
      std: 0,
      min: 0,
      max: 0,
      distribution: { low: 0, medium: 0, high: 0 }
    };
  }
  
  const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  
  // Distribution counting
  const distribution = scores.reduce(
    (dist, score) => {
      if (score <= 0.3) dist.low++;
      else if (score <= 0.6) dist.medium++;
      else dist.high++;
      return dist;
    },
    { low: 0, medium: 0, high: 0 }
  );
  
  return { mean, std, min, max, distribution };
}

/**
 * Sort nodes by entropy score
 */
export function sortNodesByEntropy(
  nodes: Node[],
  entropyScores: EntropyScores,
  ascending: boolean = true
): Node[] {
  return [...nodes].sort((a, b) => {
    const entropyA = entropyScores[a.id] || 0;
    const entropyB = entropyScores[b.id] || 0;
    
    return ascending ? entropyA - entropyB : entropyB - entropyA;
  });
}

/**
 * Filter nodes by entropy range
 */
export function filterNodesByEntropyRange(
  nodes: Node[],
  entropyScores: EntropyScores,
  minEntropy: number,
  maxEntropy: number
): Node[] {
  return nodes.filter(node => {
    const entropy = entropyScores[node.id] || 0;
    return entropy >= minEntropy && entropy <= maxEntropy;
  });
}

/**
 * Calculate dot product of two vectors
 */
function dotProduct(vector1: number[], vector2: number[]): number {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same length');
  }
  
  return vector1.reduce((sum, val, index) => sum + val * vector2[index], 0);
}

/**
 * Calculate the magnitude (length) of a vector
 */
function magnitude(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
}

/**
 * Calculate cosine similarity between two embedding vectors
 * 
 * @param vector1 - First embedding vector
 * @param vector2 - Second embedding vector
 * @returns Cosine similarity value between -1 and 1
 */
export function cosineSimilarity(vector1: number[], vector2: number[]): number {
  try {
    if (!vector1 || !vector2 || vector1.length === 0 || vector2.length === 0) {
      return 0;
    }

    if (vector1.length !== vector2.length) {
      console.warn('Vector lengths do not match');
      return 0;
    }

    const dot = dotProduct(vector1, vector2);
    const mag1 = magnitude(vector1);
    const mag2 = magnitude(vector2);

    if (mag1 === 0 || mag2 === 0) {
      return 0;
    }

    return dot / (mag1 * mag2);
  } catch (error) {
    console.error('Error calculating cosine similarity:', error);
    return 0;
  }
}

/**
 * Calculate entropy score from cosine similarity
 * Entropy = 1 - cosine_similarity
 * 
 * @param startNodeVector - Embedding vector of the start node
 * @param targetNodeVector - Embedding vector of the target node
 * @returns Entropy score between 0 and 1 (higher = more different)
 */
export function getEntropy(startNodeVector: number[], targetNodeVector: number[]): number {
  try {
    const similarity = cosineSimilarity(startNodeVector, targetNodeVector);
    const entropy = 1 - similarity;
    
    // Clamp between 0 and 1
    return Math.max(0, Math.min(1, entropy));
  } catch (error) {
    console.error('Error calculating entropy:', error);
    return 0.5; // Return neutral entropy on error
  }
}

/**
 * Calculate entropy scores for multiple target nodes
 * 
 * @param startNodeVector - Embedding vector of the start node
 * @param targetNodes - Array of {id, embedding} objects
 * @returns Object mapping node IDs to entropy scores
 */
export function calculateLocalEntropyScores(
  startNodeVector: number[], 
  targetNodes: Array<{id: string, embedding: number[]}>
): Record<string, number> {
  const entropyScores: Record<string, number> = {};
  
  for (const target of targetNodes) {
    entropyScores[target.id] = getEntropy(startNodeVector, target.embedding);
  }
  
  return entropyScores;
}

/**
 * Determine if a node should be considered a boundary based on entropy threshold
 */
export function isBoundaryNode(entropy: number, threshold: number = 0.4): boolean {
  return entropy > threshold;
}
