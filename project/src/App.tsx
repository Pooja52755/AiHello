import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import TextInput from './components/TextInput';
import GraphVisualizer from './components/GraphVisualizer';
import TraversalControl from './components/TraversalControl';
import MetricsPanel from './components/MetricsPanel';
import { GraphData, TraversalResult, ApiResponse } from './types';

function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [traversalResult, setTraversalResult] = useState<TraversalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [entropyThreshold, setEntropyThreshold] = useState(0.7);

  const handleTextProcess = useCallback(async (text: string) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/process-text', {
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
    } catch (error) {
      console.error('Error processing text:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleTraversal = useCallback(async (startingNodes: string[]) => {
    if (!graphData) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/traverse-graph', {
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

      const data: TraversalResult = await response.json();
      setTraversalResult(data);
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
          <TextInput onTextProcess={handleTextProcess} loading={loading} />
        </motion.div>

        {graphData && (
          <>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <TraversalControl
                nodes={graphData.nodes}
                selectedNodes={selectedNodes}
                onNodeSelect={handleNodeSelect}
                onTraversal={handleTraversal}
                entropyThreshold={entropyThreshold}
                onThresholdChange={setEntropyThreshold}
                loading={loading}
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
                  onNodeSelect={handleNodeSelect}
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <MetricsPanel
                  graphData={graphData}
                  traversalResult={traversalResult}
                />
              </motion.div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;