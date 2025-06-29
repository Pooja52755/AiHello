import React from 'react';
import { motion } from 'framer-motion';
import { Play, Settings, Target, Loader2, Calculator } from 'lucide-react';
import { GraphNode } from '../types';

interface TraversalControlProps {
  nodes: GraphNode[];
  selectedNodes: string[];
  onNodeSelect: (nodeId: string) => void;
  onTraversal: (startingNodes: string[]) => void;
  entropyThreshold: number;
  onThresholdChange: (threshold: number) => void;
  loading: boolean;
  onCalculateEntropy?: () => void;
  onGenerateEmbeddings?: () => void;
  embeddingsLoaded?: boolean;
  entropyScores?: { [nodeId: string]: number };
}

const TraversalControl: React.FC<TraversalControlProps> = ({
  nodes,
  selectedNodes,
  onNodeSelect,
  onTraversal,
  entropyThreshold,
  onThresholdChange,
  loading,
  onCalculateEntropy,
  onGenerateEmbeddings,
  embeddingsLoaded,
  entropyScores = {},
}) => {
  const handleTraversal = () => {
    if (selectedNodes.length === 0) {
      // Auto-select first few nodes if none selected
      const autoSelected = nodes.slice(0, 2).map(n => n.id);
      onTraversal(autoSelected);
    } else {
      onTraversal(selectedNodes);
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
      <div className="flex items-center space-x-3 mb-6">
        <Target className="h-6 w-6 text-blue-300" />
        <h3 className="text-xl font-bold text-white">Traversal Control</h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Starting Nodes</h4>
          <div className="max-h-40 overflow-y-auto space-y-2">
            {nodes.map((node) => (
              <motion.button
                key={node.id}
                onClick={() => onNodeSelect(node.id)}
                className={`w-full p-3 rounded-lg border-2 transition-all duration-200 text-left ${
                  selectedNodes.includes(node.id)
                    ? 'border-yellow-500 bg-yellow-500/20 text-yellow-100'
                    : 'border-gray-600 bg-gray-800/30 hover:bg-gray-700/40 text-gray-300'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="font-medium">
                  {node.label.length > 30 ? node.label.substring(0, 30) + '...' : node.label}
                </div>
                <div className="text-xs opacity-75">
                  Sentences: {node.sentence_ids.join(', ')}
                </div>
                {entropyScores[node.id] && (
                  <div className="text-xs text-blue-300 mt-1">
                    Entropy: {entropyScores[node.id].toFixed(3)}
                  </div>
                )}
              </motion.button>
            ))}
          </div>
          <div className="mt-3 text-sm text-blue-200">
            Selected: {selectedNodes.length} nodes
          </div>
        </div>

        <div className="space-y-6">
          <div>
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
              <Settings className="h-5 w-5" />
              <span>Configuration</span>
            </h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-blue-100 mb-2">
                  Entropy Threshold: {entropyThreshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={entropyThreshold}
                  onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Conservative (0.1)</span>
                  <span>Aggressive (1.0)</span>
                </div>
              </div>

              <div className="bg-gray-800/40 rounded-lg p-4">
                <h5 className="font-medium text-white mb-2">Algorithm Explanation</h5>
                <p className="text-sm text-gray-300">
                  Higher entropy indicates semantic divergence from the starting sentence. 
                  The algorithm stops traversal when entropy exceeds the threshold, 
                  indicating a likely sentence boundary.
                </p>
              </div>
            </div>
          </div>

          {onCalculateEntropy && (
            <motion.button
              onClick={onCalculateEntropy}
              disabled={loading}
              className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-xl shadow-lg transition-all duration-200 mb-4"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Calculator className="h-4 w-4" />
              <span>Calculate Entropy Scores</span>
            </motion.button>
          )}

          <motion.button
            onClick={handleTraversal}
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-xl shadow-lg transition-all duration-200"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {loading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Play className="h-5 w-5" />
            )}
            <span>
              {loading ? 'Traversing...' : 'Start Entropy Traversal'}
            </span>
          </motion.button>

          {onGenerateEmbeddings && (
            <motion.button
              onClick={onGenerateEmbeddings}
              disabled={loading || embeddingsLoaded}
              className={`w-full flex items-center justify-center space-x-2 px-6 py-4 rounded-xl shadow-lg transition-all duration-200 ${
                embeddingsLoaded
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : embeddingsLoaded ? (
                <span>✔️ Embeddings Loaded</span>
              ) : (
                <span>Generate Embeddings</span>
              )}
            </motion.button>
          )}
        </div>
      </div>
    </div>
  );
};

export default TraversalControl;