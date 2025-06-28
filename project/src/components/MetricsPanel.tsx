import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, TrendingUp, Target, Zap, AlertCircle } from 'lucide-react';
import { GraphData, TraversalResult } from '../types';

interface MetricsPanelProps {
  graphData: GraphData;
  traversalResult: TraversalResult | null;
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({ graphData, traversalResult }) => {
  const renderMetricCard = (title: string, value: string | number, icon: React.ReactNode, color: string) => (
    <motion.div
      className={`bg-gradient-to-br ${color} rounded-xl p-4 border border-white/10`}
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 400, damping: 10 }}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-white/80 text-sm font-medium">{title}</p>
          <p className="text-white text-2xl font-bold">{value}</p>
        </div>
        <div className="text-white/60">
          {icon}
        </div>
      </div>
    </motion.div>
  );

  const renderTraversalResults = () => {
    if (!traversalResult) return null;

    return (
      <div className="space-y-4">
        <h4 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Target className="h-5 w-5" />
          <span>Traversal Results</span>
        </h4>

        {traversalResult.traversal_results.map((result, index) => (
          <motion.div
            key={index}
            className="bg-gray-800/40 rounded-lg p-4 border border-gray-600"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-3">
              <h5 className="font-medium text-white">
                Starting Node: {result.starting_node}
              </h5>
              <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded">
                {result.stopping_reason.replace('_', ' ')}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-400">Visited Nodes:</span>
                <span className="text-white ml-2 font-medium">
                  {result.visited_nodes.length}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Boundary Nodes:</span>
                <span className="text-red-400 ml-2 font-medium">
                  {result.boundary_nodes.length}
                </span>
              </div>
            </div>

            <div className="mt-3">
              <span className="text-gray-400 text-sm">Entropy Path:</span>
              <div className="flex flex-wrap gap-1 mt-2">
                {result.entropy_path.slice(0, 5).map((step, stepIndex) => (
                  <span
                    key={stepIndex}
                    className={`text-xs px-2 py-1 rounded ${
                      step.entropy > traversalResult.threshold_used
                        ? 'bg-red-600 text-white'
                        : 'bg-green-600 text-white'
                    }`}
                  >
                    {step.entropy.toFixed(3)}
                  </span>
                ))}
                {result.entropy_path.length > 5 && (
                  <span className="text-xs text-gray-400">
                    +{result.entropy_path.length - 5} more
                  </span>
                )}
              </div>
            </div>
          </motion.div>
        ))}

        <div className="bg-gray-800/40 rounded-lg p-4 border border-gray-600">
          <h5 className="font-medium text-white mb-3">Summary Statistics</h5>
          <div className="grid grid-cols-1 gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Traversals:</span>
              <span className="text-white">{traversalResult.summary.total_traversals}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Nodes Visited:</span>
              <span className="text-white">{traversalResult.summary.avg_nodes_visited.toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Boundary Nodes:</span>
              <span className="text-white">{traversalResult.summary.avg_boundary_nodes.toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Threshold Used:</span>
              <span className="text-white">{traversalResult.threshold_used.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
        <div className="flex items-center space-x-3 mb-6">
          <BarChart className="h-6 w-6 text-blue-300" />
          <h3 className="text-xl font-bold text-white">Graph Metrics</h3>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {renderMetricCard(
            "Total Nodes",
            graphData.nodes.length,
            <Target className="h-6 w-6" />,
            "from-blue-600 to-blue-800"
          )}
          
          {renderMetricCard(
            "Total Edges",
            graphData.edges.length,
            <TrendingUp className="h-6 w-6" />,
            "from-green-600 to-green-800"
          )}
          
          {renderMetricCard(
            "Graph Density",
            (graphData.stats?.density || 0).toFixed(3),
            <Zap className="h-6 w-6" />,
            "from-purple-600 to-purple-800"
          )}
          
          {renderMetricCard(
            "Connected",
            graphData.stats?.is_connected ? "Yes" : "No",
            <AlertCircle className="h-6 w-6" />,
            graphData.stats?.is_connected 
              ? "from-green-600 to-green-800" 
              : "from-red-600 to-red-800"
          )}
        </div>
      </div>

      {traversalResult && (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
          {renderTraversalResults()}
        </div>
      )}
    </div>
  );
};

export default MetricsPanel;