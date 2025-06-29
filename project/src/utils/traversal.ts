/**
 * Graph traversal utilities with entropy-based stopping
 */

import { GraphData, Node, Edge } from '../types';
import { EntropyScores, getEntropy } from './entropy';

export interface TraversalState {
  visited: Set<string>;
  boundaryNodes: Set<string>;
  sameSentenceNodes: Set<string>;
  currentPath: string[];
  stopped: boolean;
  stoppingReason: string;
}

export interface TraversalResult {
  traversal_path: string[];
  visited_nodes: string[];
  boundary_nodes: string[];
  entropy_scores: EntropyScores;
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

/**
 * Perform entropy-guided traversal via API
 */
export async function performEntropyTraversal(
  graphData: GraphData,
  startNode: string,
  entropyThreshold: number = 0.4
): Promise<TraversalApiResponse> {
  const response = await fetch('http://localhost:5001/api/traverse-graph', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      graph: graphData,
      starting_nodes: [startNode],
      entropy_threshold: entropyThreshold,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to perform traversal');
  }

  return response.json();
}

/**
 * Build adjacency list from graph data
 */
export function buildAdjacencyList(graphData: GraphData): Map<string, string[]> {
  const adjacency = new Map<string, string[]>();
  
  // Initialize all nodes
  graphData.nodes.forEach(node => {
    adjacency.set(node.id, []);
  });
  
  // Add edges
  graphData.edges.forEach(edge => {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
    if (!adjacency.has(edge.target)) adjacency.set(edge.target, []);
    
    adjacency.get(edge.source)!.push(edge.target);
    adjacency.get(edge.target)!.push(edge.source);
  });
  
  return adjacency;
}

/**
 * Simulate local traversal (client-side) for visualization
 */
export function simulateTraversal(
  graphData: GraphData,
  startNode: string,
  entropyScores: EntropyScores,
  entropyThreshold: number,
  maxSteps: number = 20
): {
  steps: Array<{
    currentNode: string;
    neighbors: string[];
    entropy: number;
    action: 'continue' | 'boundary' | 'end';
    reason: string;
  }>;
  finalState: TraversalState;
} {
  const adjacency = buildAdjacencyList(graphData);
  const state: TraversalState = {
    visited: new Set([startNode]),
    boundaryNodes: new Set(),
    sameSentenceNodes: new Set([startNode]),
    currentPath: [startNode],
    stopped: false,
    stoppingReason: ''
  };
  
  const steps: Array<{
    currentNode: string;
    neighbors: string[];
    entropy: number;
    action: 'continue' | 'boundary' | 'end';
    reason: string;
  }> = [];
  
  const queue = [startNode];
  let stepCount = 0;
  
  while (queue.length > 0 && !state.stopped && stepCount < maxSteps) {
    const currentNode = queue.shift()!;
    const neighbors = adjacency.get(currentNode) || [];
    const unvisitedNeighbors = neighbors.filter(n => !state.visited.has(n));
    
    for (const neighbor of unvisitedNeighbors) {
      const neighborEntropy = entropyScores[neighbor] || 0;
      
      state.visited.add(neighbor);
      state.currentPath.push(neighbor);
      
      if (neighborEntropy > entropyThreshold) {
        // Boundary node detected
        state.boundaryNodes.add(neighbor);
        
        steps.push({
          currentNode: neighbor,
          neighbors: adjacency.get(neighbor) || [],
          entropy: neighborEntropy,
          action: 'boundary',
          reason: `Entropy ${neighborEntropy.toFixed(3)} > threshold ${entropyThreshold}`
        });
        
        // Don't continue from boundary nodes
        continue;
      } else {
        // Continue traversal
        state.sameSentenceNodes.add(neighbor);
        queue.push(neighbor);
        
        steps.push({
          currentNode: neighbor,
          neighbors: adjacency.get(neighbor) || [],
          entropy: neighborEntropy,
          action: 'continue',
          reason: `Entropy ${neighborEntropy.toFixed(3)} ≤ threshold ${entropyThreshold}`
        });
      }
      
      stepCount++;
      if (stepCount >= maxSteps) {
        state.stopped = true;
        state.stoppingReason = 'Maximum steps reached';
        break;
      }
    }
    
    if (queue.length === 0) {
      state.stopped = true;
      state.stoppingReason = 'No more nodes to explore';
    }
  }
  
  return { steps, finalState: state };
}

/**
 * Calculate traversal efficiency metrics
 */
export function calculateTraversalEfficiency(
  traversalResult: TraversalResult,
  totalNodes: number
): {
  coverageRatio: number;
  boundaryRatio: number;
  pathEfficiency: number;
  stoppingAccuracy: number;
} {
  const visitedCount = traversalResult.visited_nodes.length;
  const boundaryCount = traversalResult.boundary_nodes.length;
  const pathLength = traversalResult.traversal_path.length;
  
  return {
    coverageRatio: visitedCount / totalNodes,
    boundaryRatio: boundaryCount / totalNodes,
    pathEfficiency: visitedCount / pathLength, // How many unique nodes per step
    stoppingAccuracy: boundaryCount / (boundaryCount + visitedCount) // Precision of boundary detection
  };
}

/**
 * Get node status based on traversal results
 */
export function getNodeStatus(
  nodeId: string,
  traversalResult: TraversalResult | null,
  selectedNodes: string[]
): {
  isSelected: boolean;
  isVisited: boolean;
  isBoundary: boolean;
  isInPath: boolean;
  pathIndex: number;
} {
  const isSelected = selectedNodes.includes(nodeId);
  
  if (!traversalResult) {
    return {
      isSelected,
      isVisited: false,
      isBoundary: false,
      isInPath: false,
      pathIndex: -1
    };
  }
  
  const isVisited = traversalResult.visited_nodes.includes(nodeId);
  const isBoundary = traversalResult.boundary_nodes.includes(nodeId);
  const pathIndex = traversalResult.traversal_path.indexOf(nodeId);
  const isInPath = pathIndex >= 0;
  
  return {
    isSelected,
    isVisited,
    isBoundary,
    isInPath,
    pathIndex
  };
}

/**
 * Generate color for traversal visualization
 */
export function getTraversalColor(
  nodeId: string,
  traversalResult: TraversalResult | null,
  selectedNodes: string[],
  entropyScores: EntropyScores,
  entropyThreshold: number
): string {
  const status = getNodeStatus(nodeId, traversalResult, selectedNodes);
  
  if (status.isSelected) return '#8B5CF6'; // Purple for selected
  if (status.isBoundary) return '#EF4444';  // Red for boundary
  if (status.isVisited) return '#10B981';   // Green for visited (same sentence)
  
  // Color by entropy if available
  if (nodeId in entropyScores) {
    const entropy = entropyScores[nodeId];
    if (entropy > entropyThreshold) return '#F59E0B'; // Orange for high entropy
    if (entropy < entropyThreshold * 0.5) return '#3B82F6'; // Blue for low entropy
  }
  
  return '#6B7280'; // Gray for default
}

/**
 * Get traversal animation delay for path visualization
 */
export function getTraversalAnimationDelay(pathIndex: number): number {
  return pathIndex * 200; // 200ms between each step
}

/**
 * Enhanced entropy-based BFS traversal with embeddings
 */
export function entropyBasedBFS(
  graphData: GraphData,
  startNode: string,
  nodeEmbeddings: Record<string, number[]>,
  entropyThreshold: number = 0.4,
  maxNodes: number = 50
): TraversalResult {
  const adjacency = buildAdjacencyList(graphData);
  const visited = new Set<string>();
  const boundaryNodes = new Set<string>();
  const sameSentenceNodes = new Set<string>([startNode]);
  const traversalPath: string[] = [];
  const entropyScores: EntropyScores = {};
  const boundaryPredictions: { [nodeId: string]: boolean } = {};
  
  if (!nodeEmbeddings[startNode]) {
    throw new Error(`No embedding found for start node: ${startNode}`);
  }
  
  const startEmbedding = nodeEmbeddings[startNode];
  const queue = [startNode];
  visited.add(startNode);
  traversalPath.push(startNode);
  entropyScores[startNode] = 0; // Start node has 0 entropy relative to itself
  boundaryPredictions[startNode] = false;
  
  while (queue.length > 0 && visited.size < maxNodes) {
    const currentNode = queue.shift()!;
    const neighbors = adjacency.get(currentNode) || [];
    
    for (const neighbor of neighbors) {
      if (visited.has(neighbor) || !nodeEmbeddings[neighbor]) continue;
      
      // Calculate entropy between start node and neighbor
      const neighborEmbedding = nodeEmbeddings[neighbor];
      const entropy = getEntropy(startEmbedding, neighborEmbedding);
      
      visited.add(neighbor);
      traversalPath.push(neighbor);
      entropyScores[neighbor] = entropy;
      
      if (entropy > entropyThreshold) {
        // High entropy indicates sentence boundary
        boundaryNodes.add(neighbor);
        boundaryPredictions[neighbor] = true;
        // Don't continue from boundary nodes
      } else {
        // Low entropy indicates same sentence
        sameSentenceNodes.add(neighbor);
        boundaryPredictions[neighbor] = false;
        queue.push(neighbor);
      }
    }
  }
  
  return {
    traversal_path: traversalPath,
    visited_nodes: Array.from(visited),
    boundary_nodes: Array.from(boundaryNodes),
    entropy_scores: entropyScores,
    boundary_predictions: boundaryPredictions,
    metrics: {
      total_nodes: graphData.nodes.length,
      visited_count: visited.size,
      boundary_count: boundaryNodes.size,
      efficiency: visited.size / traversalPath.length,
      entropy_threshold: entropyThreshold
    }
  };
}

/**
 * Enhanced entropy-based DFS traversal with embeddings
 */
export function entropyBasedDFS(
  graphData: GraphData,
  startNode: string,
  nodeEmbeddings: Record<string, number[]>,
  entropyThreshold: number = 0.4,
  maxDepth: number = 10
): TraversalResult {
  const adjacency = buildAdjacencyList(graphData);
  const visited = new Set<string>();
  const boundaryNodes = new Set<string>();
  const sameSentenceNodes = new Set<string>([startNode]);
  const traversalPath: string[] = [];
  const entropyScores: EntropyScores = {};
  const boundaryPredictions: { [nodeId: string]: boolean } = {};
  
  if (!nodeEmbeddings[startNode]) {
    throw new Error(`No embedding found for start node: ${startNode}`);
  }
  
  const startEmbedding = nodeEmbeddings[startNode];
  
  function dfsRecursive(currentNode: string, depth: number): void {
    if (depth >= maxDepth || visited.has(currentNode)) return;
    
    visited.add(currentNode);
    traversalPath.push(currentNode);
    
    if (currentNode === startNode) {
      entropyScores[currentNode] = 0;
      boundaryPredictions[currentNode] = false;
    } else if (nodeEmbeddings[currentNode]) {
      const entropy = getEntropy(startEmbedding, nodeEmbeddings[currentNode]);
      entropyScores[currentNode] = entropy;
      
      if (entropy > entropyThreshold) {
        boundaryNodes.add(currentNode);
        boundaryPredictions[currentNode] = true;
        return; // Stop traversing from boundary nodes
      } else {
        sameSentenceNodes.add(currentNode);
        boundaryPredictions[currentNode] = false;
      }
    }
    
    const neighbors = adjacency.get(currentNode) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        dfsRecursive(neighbor, depth + 1);
      }
    }
  }
  
  dfsRecursive(startNode, 0);
  
  return {
    traversal_path: traversalPath,
    visited_nodes: Array.from(visited),
    boundary_nodes: Array.from(boundaryNodes),
    entropy_scores: entropyScores,
    boundary_predictions: boundaryPredictions,
    metrics: {
      total_nodes: graphData.nodes.length,
      visited_count: visited.size,
      boundary_count: boundaryNodes.size,
      efficiency: visited.size / traversalPath.length,
      entropy_threshold: entropyThreshold
    }
  };
}

/**
 * Adaptive traversal that chooses BFS or DFS based on graph structure
 */
export function adaptiveEntropyTraversal(
  graphData: GraphData,
  startNode: string,
  nodeEmbeddings: Record<string, number[]>,
  entropyThreshold: number = 0.4
): TraversalResult {
  const avgDegree = graphData.edges.length * 2 / graphData.nodes.length;
  
  // Use BFS for dense graphs, DFS for sparse graphs
  if (avgDegree > 4) {
    return entropyBasedBFS(graphData, startNode, nodeEmbeddings, entropyThreshold);
  } else {
    return entropyBasedDFS(graphData, startNode, nodeEmbeddings, entropyThreshold);
  }
}

/**
 * Multi-start traversal for comprehensive boundary detection
 */
export function multiStartEntropyTraversal(
  graphData: GraphData,
  startNodes: string[],
  nodeEmbeddings: Record<string, number[]>,
  entropyThreshold: number = 0.4
): {
  results: TraversalResult[];
  aggregated: TraversalResult;
  consensus: { [nodeId: string]: number }; // Consensus boundary probability
} {
  const results: TraversalResult[] = [];
  const allVisited = new Set<string>();
  const allBoundaries = new Set<string>();
  const allPaths: string[] = [];
  const aggregatedScores: EntropyScores = {};
  const consensusScores: { [nodeId: string]: number } = {};
  
  // Run traversal from each start node
  for (const startNode of startNodes) {
    try {
      const result = entropyBasedBFS(graphData, startNode, nodeEmbeddings, entropyThreshold);
      results.push(result);
      
      // Aggregate results
      result.visited_nodes.forEach(node => allVisited.add(node));
      result.boundary_nodes.forEach(node => allBoundaries.add(node));
      allPaths.push(...result.traversal_path);
      
      // Merge entropy scores (take average)
      Object.entries(result.entropy_scores).forEach(([node, score]) => {
        if (aggregatedScores[node]) {
          aggregatedScores[node] = (aggregatedScores[node] + score) / 2;
        } else {
          aggregatedScores[node] = score;
        }
      });
    } catch (error) {
      console.warn(`Failed to traverse from node ${startNode}:`, error);
    }
  }
  
  // Calculate consensus boundary probabilities
  graphData.nodes.forEach(node => {
    const boundaryCount = results.reduce((count, result) => 
      count + (result.boundary_predictions[node.id] ? 1 : 0), 0
    );
    consensusScores[node.id] = boundaryCount / results.length;
  });
  
  const aggregated: TraversalResult = {
    traversal_path: Array.from(new Set(allPaths)), // Remove duplicates
    visited_nodes: Array.from(allVisited),
    boundary_nodes: Array.from(allBoundaries),
    entropy_scores: aggregatedScores,
    boundary_predictions: Object.fromEntries(
      Object.entries(consensusScores).map(([node, prob]) => [node, prob > 0.5])
    ),
    metrics: {
      total_nodes: graphData.nodes.length,
      visited_count: allVisited.size,
      boundary_count: allBoundaries.size,
      efficiency: allVisited.size / allPaths.length,
      entropy_threshold: entropyThreshold
    }
  };
  
  return { results, aggregated, consensus: consensusScores };
}
