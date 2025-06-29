import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import TextInput from './components/TextInput';
import GraphVisualizer from './components/GraphVisualizer';
import TraversalControl from './components/TraversalControl';
import MetricsPanel from './components/MetricsPanel';
import EvaluationPanel from './components/EvaluationPanel';
import { GraphData, TraversalResult, ApiResponse } from './types';

function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [traversalResult, setTraversalResult] = useState<TraversalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [entropyThreshold, setEntropyThreshold] = useState(0.7);
  const [entropyScores, setEntropyScores] = useState<{ [nodeId: string]: number }>({});
  const [nodeEmbeddings, setNodeEmbeddings] = useState<{ [nodeId: string]: number[] }>({});
  const [embeddingsLoaded, setEmbeddingsLoaded] = useState(false);

  const [traversalApiResponse, setTraversalApiResponse] = useState<any | null>(null);

  const calculateEntropyScores = useCallback(async () => {
    if (!graphData) return;

    try {
      const response = await fetch('http://localhost:5001/api/calculate-entropy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ graph: graphData }),
      });

      if (!response.ok) {
        throw new Error('Failed to calculate entropy');
      }

      const data = await response.json();
      setEntropyScores(data.entropy_scores || {});
    } catch (error) {
      console.error('Error calculating entropy:', error);
      setEntropyScores({});
    }
  }, [graphData]);

  const handleTextProcess = useCallback(async (text: string) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5001/api/process-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to process text');
      }

      const data: ApiResponse = await response.json();
      setGraphData(data.graph);
      setSelectedNodes([]);
      setTraversalResult(null);
      setEntropyScores({});
      
      // Auto-calculate entropy scores after processing text
      setTimeout(() => {
        calculateEntropyScores();
      }, 500);
    } catch (error) {
      console.error('Error processing text:', error);
    } finally {
      setLoading(false);
    }
  }, [calculateEntropyScores]);

  const handleTraversal = useCallback(async (startingNodes: string[]) => {
    if (!graphData) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5001/api/traverse-graph', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          graph: graphData,
          starting_nodes: startingNodes,
          entropy_threshold: entropyThreshold,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to traverse graph');
      }

      const data = await response.json();
      
      // Store the full API response
      setTraversalApiResponse(data);
      
      // Extract traversal result for the main UI
      setTraversalResult(data.traversal_result);
      
      // Update entropy scores if returned
      if (data.traversal_result?.entropy_scores) {
        setEntropyScores(data.traversal_result.entropy_scores);
      }
    } catch (error) {
      console.error('Error traversing graph:', error);
    } finally {
      setLoading(false);
    }
  }, [graphData, entropyThreshold]);

  const handleNodeSelect = useCallback((nodeId: string) => {
    setSelectedNodes(prev => {
      if (prev.includes(nodeId)) {
        return prev.filter(id => id !== nodeId);
      } else {
        return [...prev, nodeId];
      }
    });
  }, []);

  // Generate and load node embeddings
  const generateEmbeddings = useCallback(async () => {
    if (!graphData) return;

    try {
      setLoading(true);
      const response = await fetch('http://localhost:5001/api/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          nodes: graphData.nodes.map(node => ({
            id: node.id,
            text: node.label
          }))
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate embeddings');
      }

      const data = await response.json();
      setNodeEmbeddings(data.embeddings || {});
      setEmbeddingsLoaded(true);
    } catch (error) {
      console.error('Error generating embeddings:', error);
      setEmbeddingsLoaded(false);
    } finally {
      setLoading(false);
    }
  }, [graphData]);

  // Helper function for local entropy calculation
  const calculateLocalEntropy = (embedding1: number[], embedding2: number[]): number => {
    try {
      // Calculate cosine similarity
      const dotProduct = embedding1.reduce((sum, val, idx) => sum + val * embedding2[idx], 0);
      const magnitude1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val * val, 0));
      const magnitude2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val * val, 0));
      
      if (magnitude1 === 0 || magnitude2 === 0) return 0.5;
      
      const similarity = dotProduct / (magnitude1 * magnitude2);
      return Math.max(0, Math.min(1, 1 - similarity)); // Convert to entropy (1 - similarity)
    } catch (error) {
      console.error('Error calculating local entropy:', error);
      return 0.5;
    }
  };

  // Enhanced text processing with embedding generation
  const handleEnhancedTextProcess = useCallback(async (text: string) => {
    await handleTextProcess(text);
    // Auto-generate embeddings after processing
    setTimeout(generateEmbeddings, 1000);
  }, [handleTextProcess, generateEmbeddings]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      <Toaster 
        position="top-right"
        toastOptions={{
          style: {
            background: '#1f2937',
            color: '#f9fafb',
            border: '1px solid #374151',
          },
        }}
      />
      
      <Header />
      
      <main className="container mx-auto px-4 py-8 space-y-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <TextInput onTextProcess={handleEnhancedTextProcess} loading={loading} />
        </motion.div>

        {graphData && (
          <>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >                <TraversalControl
                  nodes={graphData.nodes}
                  selectedNodes={selectedNodes}
                  onNodeSelect={handleNodeSelect}
                  onTraversal={handleTraversal}
                  entropyThreshold={entropyThreshold}
                  onThresholdChange={setEntropyThreshold}
                  loading={loading}
                  onCalculateEntropy={calculateEntropyScores}
                  onGenerateEmbeddings={generateEmbeddings}
                  embeddingsLoaded={embeddingsLoaded}
                  entropyScores={entropyScores}
                />
            </motion.div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
              <motion.div
                className="xl:col-span-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                <GraphVisualizer
                  graphData={graphData}
                  traversalResult={traversalResult}
                  selectedNodes={selectedNodes}
                  entropyScores={entropyScores}
                  entropyThreshold={entropyThreshold}
                  onNodeSelect={handleNodeSelect}
                />
              </motion.div>

              <motion.div
                className="space-y-6"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <MetricsPanel
                  graphData={graphData}
                  traversalResult={traversalResult}
                />
                
                {traversalApiResponse && (
                  <EvaluationPanel
                    evaluationMetrics={traversalApiResponse.evaluation_metrics || null}
                    traversalResult={traversalResult}
                    entropyScores={entropyScores}
                  />
                )}
              </motion.div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;