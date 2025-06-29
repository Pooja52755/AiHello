import React from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Target, Activity, TrendingUp, Info } from 'lucide-react';
import { EvaluationMetrics, TraversalResult } from '../utils/traversal';
import { EntropyScores, calculateEntropyStats } from '../utils/entropy';

interface EvaluationPanelProps {
  evaluationMetrics: EvaluationMetrics | null;
  traversalResult: TraversalResult | null;
  entropyScores: EntropyScores;
  className?: string;
}

const EvaluationPanel: React.FC<EvaluationPanelProps> = ({
  evaluationMetrics,
  traversalResult,
  entropyScores,
  className = ''
}) => {
  const entropyStats = calculateEntropyStats(entropyScores);

  const MetricCard: React.FC<{
    icon: React.ElementType;
    title: string;
    value: string | number;
    description: string;
    color: string;
  }> = ({ icon: Icon, title, value, description, color }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white rounded-lg border border-gray-200 p-4 shadow-sm ${color}`}
    >
      <div className="flex items-center justify-between mb-2">
        <Icon className="w-5 h-5 text-gray-600" />
        <span className="text-2xl font-bold text-gray-800">
          {typeof value === 'number' ? value.toFixed(3) : value}
        </span>
      </div>
      <h3 className="font-semibold text-gray-700 mb-1">{title}</h3>
      <p className="text-sm text-gray-500">{description}</p>
    </motion.div>
  );

  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Boundary Detection Metrics */}
      {evaluationMetrics && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <Target className="w-5 h-5" />
            Boundary Detection Performance
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <MetricCard
              icon={BarChart3}
              title="F1-Score"
              value={evaluationMetrics.f1_score}
              description="Harmonic mean of precision and recall"
              color="border-l-4 border-l-blue-500"
            />
            
            <MetricCard
              icon={TrendingUp}
              title="Precision"
              value={evaluationMetrics.precision}
              description="Accuracy of boundary predictions"
              color="border-l-4 border-l-green-500"
            />
            
            <MetricCard
              icon={Activity}
              title="Recall"
              value={evaluationMetrics.recall}
              description="Coverage of actual boundaries"
              color="border-l-4 border-l-yellow-500"
            />
            
            <MetricCard
              icon={Target}
              title="Accuracy"
              value={evaluationMetrics.accuracy}
              description="Overall classification accuracy"
              color="border-l-4 border-l-purple-500"
            />
            
            <MetricCard
              icon={Info}
              title="True Positives"
              value={evaluationMetrics.true_positives}
              description="Correctly identified boundaries"
              color="border-l-4 border-l-emerald-500"
            />
            
            <MetricCard
              icon={Info}
              title="False Positives"
              value={evaluationMetrics.false_positives}
              description="Incorrectly identified boundaries"
              color="border-l-4 border-l-red-500"
            />
          </div>
        </motion.section>
      )}

      {/* Traversal Efficiency */}
      {traversalResult && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Traversal Efficiency
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              icon={BarChart3}
              title="Nodes Visited"
              value={traversalResult.metrics.visited_count}
              description={`Out of ${traversalResult.metrics.total_nodes} total nodes`}
              color="border-l-4 border-l-blue-500"
            />
            
            <MetricCard
              icon={Target}
              title="Boundaries Found"
              value={traversalResult.metrics.boundary_count}
              description="Detected sentence boundaries"
              color="border-l-4 border-l-red-500"
            />
            
            <MetricCard
              icon={TrendingUp}
              title="Efficiency"
              value={formatPercentage(traversalResult.metrics.efficiency)}
              description="Percentage of graph explored"
              color="border-l-4 border-l-green-500"
            />
            
            <MetricCard
              icon={Activity}
              title="Threshold"
              value={traversalResult.metrics.entropy_threshold}
              description="Entropy stopping threshold"
              color="border-l-4 border-l-purple-500"
            />
          </div>
        </motion.section>
      )}

      {/* Entropy Statistics */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="space-y-4"
      >
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Entropy Analysis
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <MetricCard
            icon={Activity}
            title="Mean Entropy"
            value={entropyStats.mean}
            description="Average entropy across all nodes"
            color="border-l-4 border-l-blue-500"
          />
          
          <MetricCard
            icon={TrendingUp}
            title="Std Deviation"
            value={entropyStats.std}
            description="Entropy variation in the graph"
            color="border-l-4 border-l-green-500"
          />
          
          <MetricCard
            icon={BarChart3}
            title="Entropy Range"
            value={`${entropyStats.min.toFixed(3)} - ${entropyStats.max.toFixed(3)}`}
            description="Min and max entropy values"
            color="border-l-4 border-l-yellow-500"
          />
        </div>

        {/* Entropy Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm"
        >
          <h3 className="font-semibold text-gray-700 mb-4">Entropy Distribution</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Low Entropy (≤ 0.3)</span>
              <div className="flex items-center gap-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{
                      width: `${(entropyStats.distribution.low / Object.keys(entropyScores).length) * 100}%`
                    }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-800">
                  {entropyStats.distribution.low}
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Medium Entropy (0.3 - 0.6)</span>
              <div className="flex items-center gap-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-yellow-500 h-2 rounded-full"
                    style={{
                      width: `${(entropyStats.distribution.medium / Object.keys(entropyScores).length) * 100}%`
                    }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-800">
                  {entropyStats.distribution.medium}
                </span>
              </div>
            </div>              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">High Entropy (&gt; 0.6)</span>
              <div className="flex items-center gap-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-red-500 h-2 rounded-full"
                    style={{
                      width: `${(entropyStats.distribution.high / Object.keys(entropyScores).length) * 100}%`
                    }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-800">
                  {entropyStats.distribution.high}
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.section>

      {/* Legend */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm"
      >
        <h3 className="font-semibold text-gray-700 mb-3">Node Color Legend</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-purple-500"></div>
            <span>Selected</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span>Same Sentence</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span>Boundary</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500"></div>
            <span>Low Entropy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
            <span>Medium Entropy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-gray-500"></div>
            <span>Default</span>
          </div>
        </div>
      </motion.section>
    </div>
  );
};

export default EvaluationPanel;
