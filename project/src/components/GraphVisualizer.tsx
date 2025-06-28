import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Network, Eye, Maximize2 } from 'lucide-react';
import { GraphData, TraversalResult } from '../types';

interface GraphVisualizerProps {
  graphData: GraphData;
  traversalResult: TraversalResult | null;
  selectedNodes: string[];
  onNodeSelect: (nodeId: string) => void;
}

const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  graphData,
  traversalResult,
  selectedNodes,
  onNodeSelect,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current) {
        const rect = svgRef.current.parentElement?.getBoundingClientRect();
        if (rect) {
          setDimensions({
            width: rect.width - 32,
            height: Math.max(600, rect.height - 32),
          });
        }
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Simple force-directed layout for nodes
  const layoutNodes = () => {
    const nodes = graphData.nodes.map(node => ({
      ...node,
      x: Math.random() * (dimensions.width - 100) + 50,
      y: Math.random() * (dimensions.height - 100) + 50,
    }));

    // Simple spring layout simulation
    for (let i = 0; i < 100; i++) {
      nodes.forEach(node => {
        let fx = 0, fy = 0;

        // Repulsion from other nodes
        nodes.forEach(other => {
          if (node.id !== other.id) {
            const dx = node.x - other.x;
            const dy = node.y - other.y;
            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = 1000 / (distance * distance);
            fx += (dx / distance) * force;
            fy += (dy / distance) * force;
          }
        });

        // Attraction from connected nodes
        graphData.edges.forEach(edge => {
          if (edge.source === node.id) {
            const target = nodes.find(n => n.id === edge.target);
            if (target) {
              const dx = target.x - node.x;
              const dy = target.y - node.y;
              const distance = Math.sqrt(dx * dx + dy * dy) || 1;
              const force = distance * 0.01;
              fx += (dx / distance) * force;
              fy += (dy / distance) * force;
            }
          }
          if (edge.target === node.id) {
            const source = nodes.find(n => n.id === edge.source);
            if (source) {
              const dx = source.x - node.x;
              const dy = source.y - node.y;
              const distance = Math.sqrt(dx * dx + dy * dy) || 1;
              const force = distance * 0.01;
              fx += (dx / distance) * force;
              fy += (dy / distance) * force;
            }
          }
        });

        // Update position
        node.x += fx * 0.01;
        node.y += fy * 0.01;

        // Keep within bounds
        node.x = Math.max(50, Math.min(dimensions.width - 50, node.x));
        node.y = Math.max(50, Math.min(dimensions.height - 50, node.y));
      });
    }

    return nodes;
  };

  const layoutedNodes = layoutNodes();

  const getNodeColor = (nodeId: string) => {
    if (selectedNodes.includes(nodeId)) return '#F59E0B';
    
    if (traversalResult) {
      const isVisited = traversalResult.traversal_results.some(result =>
        result.visited_nodes.includes(nodeId)
      );
      const isBoundary = traversalResult.traversal_results.some(result =>
        result.boundary_nodes.includes(nodeId)
      );
      
      if (isBoundary) return '#EF4444';
      if (isVisited) return '#10B981';
    }
    
    return '#3B82F6';
  };

  const getNodeOpacity = (nodeId: string) => {
    if (!traversalResult) return 1;
    
    const isInvolved = traversalResult.traversal_results.some(result =>
      result.visited_nodes.includes(nodeId) || result.boundary_nodes.includes(nodeId)
    );
    
    return isInvolved ? 1 : 0.3;
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Network className="h-6 w-6 text-blue-300" />
          <h3 className="text-xl font-bold text-white">Knowledge Graph Visualization</h3>
        </div>
        <div className="flex items-center space-x-2">
          <Eye className="h-5 w-5 text-blue-300" />
          <Maximize2 className="h-5 w-5 text-blue-300" />
        </div>
      </div>

      <div className="bg-gray-900/50 rounded-xl p-4 mb-4">
        <div className="flex flex-wrap gap-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-gray-300">Default Nodes</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span className="text-gray-300">Selected</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-gray-300">Visited</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-gray-300">Boundary</span>
          </div>
        </div>
      </div>

      <div className="relative overflow-hidden rounded-xl bg-gray-900/30">
        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          className="w-full"
        >
          {/* Edges */}
          {graphData.edges.map((edge, index) => {
            const sourceNode = layoutedNodes.find(n => n.id === edge.source);
            const targetNode = layoutedNodes.find(n => n.id === edge.target);
            
            if (!sourceNode || !targetNode) return null;

            return (
              <motion.g key={index} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <line
                  x1={sourceNode.x}
                  y1={sourceNode.y}
                  x2={targetNode.x}
                  y2={targetNode.y}
                  stroke="#6B7280"
                  strokeWidth="2"
                  strokeOpacity="0.6"
                  markerEnd="url(#arrowhead)"
                />
                <text
                  x={(sourceNode.x + targetNode.x) / 2}
                  y={(sourceNode.y + targetNode.y) / 2}
                  fill="#9CA3AF"
                  fontSize="10"
                  textAnchor="middle"
                  className="pointer-events-none"
                >
                  {edge.label}
                </text>
              </motion.g>
            );
          })}

          {/* Nodes */}
          {layoutedNodes.map((node, index) => (
            <motion.g
              key={node.id}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ 
                opacity: getNodeOpacity(node.id), 
                scale: 1 
              }}
              transition={{ delay: index * 0.05 }}
              style={{ cursor: 'pointer' }}
              onClick={() => onNodeSelect(node.id)}
            >
              <circle
                cx={node.x}
                cy={node.y}
                r="20"
                fill={getNodeColor(node.id)}
                stroke="#FFFFFF"
                strokeWidth="2"
                className="hover:stroke-4 transition-all duration-200"
              />
              <text
                x={node.x}
                y={node.y - 25}
                fill="#FFFFFF"
                fontSize="12"
                textAnchor="middle"
                className="pointer-events-none font-medium"
              >
                {node.label.length > 15 ? node.label.substring(0, 15) + '...' : node.label}
              </text>
            </motion.g>
          ))}

          {/* Arrow marker definition */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill="#6B7280"
              />
            </marker>
          </defs>
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="text-blue-300 font-medium">Nodes</div>
          <div className="text-white text-lg">{graphData.nodes.length}</div>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="text-blue-300 font-medium">Edges</div>
          <div className="text-white text-lg">{graphData.edges.length}</div>
        </div>
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="text-blue-300 font-medium">Density</div>
          <div className="text-white text-lg">{(graphData.stats?.density || 0).toFixed(3)}</div>
        </div>
      </div>
    </div>
  );
};

export default GraphVisualizer;