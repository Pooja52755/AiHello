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

        <motion.div
          className="bg-gray-800/40 rounded-lg p-4 border border-gray-600"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between mb-3">
            <h5 className="font-medium text-white">
              Entropy-Based Traversal
            </h5>
            <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded">
              Complete
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-gray-400">Visited Nodes:</span>
              <span className="text-white ml-2 font-medium">
                {traversalResult.visited_nodes?.length || 0}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Boundary Nodes:</span>
              <span className="text-red-400 ml-2 font-medium">
                {traversalResult.boundary_nodes?.length || 0}
              </span>
            </div>
          </div>

          <div className="mt-3">
            <span className="text-gray-400 text-sm">Traversal Path:</span>
            <div className="flex flex-wrap gap-1 mt-2">
              {traversalResult.traversal_path?.slice(0, 8).map((nodeId, stepIndex) => (
                <span
                  key={stepIndex}
                  className={`text-xs px-2 py-1 rounded ${
                    traversalResult.boundary_nodes?.includes(nodeId)
                      ? 'bg-red-600 text-white'
                      : 'bg-green-600 text-white'
                  }`}
                >
                  {nodeId.length > 8 ? nodeId.substring(0, 8) + '...' : nodeId}
                </span>
              ))}
              {traversalResult.traversal_path && traversalResult.traversal_path.length > 8 && (
                <span className="text-xs px-2 py-1 rounded bg-gray-600 text-white">
                  +{traversalResult.traversal_path.length - 8} more
                </span>
              )}
            </div>
          </div>

          {/* Efficiency Metrics */}
          <div className="mt-4 pt-3 border-t border-gray-600">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-400">Efficiency:</span>
                <span className="text-white ml-2 font-medium">
                  {traversalResult.metrics?.efficiency ? (traversalResult.metrics.efficiency * 100).toFixed(1) : '0'}%
                </span>
              </div>
              <div>
                <span className="text-gray-400">Threshold:</span>
                <span className="text-blue-400 ml-2 font-medium">
                  {traversalResult.metrics?.entropy_threshold?.toFixed(2) || 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    );
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
      <div className="flex items-center space-x-3 mb-6">
        <BarChart className="h-6 w-6 text-blue-300" />
        <h3 className="text-xl font-bold text-white">Graph Metrics</h3>
      </div>

      <div className="space-y-6">
        {/* Basic Graph Metrics */}
        <div className="grid grid-cols-2 gap-4">
          {renderMetricCard(
            "Total Nodes",
            graphData.nodes.length,
            <TrendingUp className="h-5 w-5" />,
            "from-blue-600 to-blue-700"
          )}
          {renderMetricCard(
            "Total Edges",
            graphData.edges.length,
            <Target className="h-5 w-5" />,
            "from-green-600 to-green-700"
          )}
        </div>

        {/* Performance Metrics */}
        {traversalResult && (
          <div className="grid grid-cols-2 gap-4">
            {renderMetricCard(
              "Visited Nodes",
              traversalResult.visited_nodes?.length || 0,
              <Zap className="h-5 w-5" />,
              "from-purple-600 to-purple-700"
            )}
            {renderMetricCard(
              "Boundary Nodes",
              traversalResult.boundary_nodes?.length || 0,
              <AlertCircle className="h-5 w-5" />,
              "from-red-600 to-red-700"
            )}
          </div>
        )}

        {/* Traversal Results */}
        {renderTraversalResults()}

        {/* Graph Statistics */}
        <div className="bg-gray-800/40 rounded-lg p-4 border border-gray-600">
          <h4 className="text-lg font-semibold text-white mb-3">Graph Statistics</h4>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Node Degree:</span>
              <span className="text-white">
                {graphData.edges.length > 0 ? (graphData.edges.length * 2 / graphData.nodes.length).toFixed(1) : '0'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Graph Density:</span>
              <span className="text-white">
                {graphData.nodes.length > 1 ? 
                  ((graphData.edges.length / (graphData.nodes.length * (graphData.nodes.length - 1) / 2)) * 100).toFixed(1) + '%' : 
                  '0%'
                }
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsPanel;