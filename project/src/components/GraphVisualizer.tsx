import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Network, Eye, ZoomIn, ZoomOut } from 'lucide-react';
import { GraphData, TraversalResult } from '../types';

interface GraphVisualizerProps {
  graphData: GraphData;
  traversalResult: TraversalResult | null;
  selectedNodes: string[];
  entropyScores: { [nodeId: string]: number };
  entropyThreshold: number;
  onNodeSelect: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null, nodeData?: any) => void;
}

interface TooltipData {
  nodeId: string;
  label: string;
  entropy: number;
  status: string;
  x: number;
  y: number;
}

const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  graphData,
  traversalResult,
  selectedNodes,
  entropyScores,
  entropyThreshold,
  onNodeSelect,
  onNodeHover
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

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

  // Zoom controls
  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.2, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.2, 0.5));
  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

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

  // Enhanced node color based on entropy and traversal state
  const getNodeColor = useCallback((nodeId: string) => {
    // Selected nodes
    if (selectedNodes.includes(nodeId)) return '#8B5CF6'; // Purple
    
    // Traversal-based coloring
    if (traversalResult) {
      if (traversalResult.boundary_nodes.includes(nodeId)) return '#EF4444'; // Red for boundary
      if (traversalResult.visited_nodes.includes(nodeId)) return '#10B981'; // Green for visited
    }
    
    // Entropy-based coloring
    const entropy = entropyScores[nodeId] || 0;
    if (entropy > entropyThreshold) return '#F59E0B'; // Orange for high entropy
    if (entropy < entropyThreshold * 0.5) return '#3B82F6'; // Blue for low entropy
    
    return '#6B7280'; // Gray for default
  }, [selectedNodes, traversalResult, entropyScores, entropyThreshold]);

  // Enhanced node size based on importance/entropy
  const getNodeSize = useCallback((nodeId: string) => {
    const baseSize = 8;
    const entropy = entropyScores[nodeId] || 0;
    
    if (selectedNodes.includes(nodeId)) return baseSize + 6;
    if (traversalResult?.boundary_nodes.includes(nodeId)) return baseSize + 4;
    if (traversalResult?.visited_nodes.includes(nodeId)) return baseSize + 2;
    
    // Size based on entropy (higher entropy = larger)
    return baseSize + (entropy * 4);
  }, [entropyScores, traversalResult, selectedNodes]);

  // Handle node click with entropy information
  const handleNodeClick = useCallback((nodeId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    onNodeSelect(nodeId);
    
    if (onNodeHover) {
      const entropy = entropyScores[nodeId] || 0;
      onNodeHover(nodeId, { entropy });
    }
  }, [onNodeSelect, onNodeHover, entropyScores]);

  // Handle node hover for tooltip
  const handleNodeHover = useCallback((nodeId: string, event: React.MouseEvent) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (rect) {
      const node = graphData.nodes.find(n => n.id === nodeId);
      const entropy = entropyScores[nodeId] || 0;
      
      let statusText = 'Default';
      if (selectedNodes.includes(nodeId)) statusText = 'Selected';
      else if (traversalResult?.boundary_nodes.includes(nodeId)) statusText = 'Boundary';
      else if (traversalResult?.visited_nodes.includes(nodeId)) statusText = 'Same Sentence';
      else if (entropy > entropyThreshold) statusText = 'High Entropy';
      else if (entropy < entropyThreshold * 0.5) statusText = 'Low Entropy';
      
      setTooltip({
        nodeId,
        label: node?.label || nodeId,
        entropy,
        status: statusText,
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      });
    }
  }, [graphData.nodes, entropyScores, traversalResult, selectedNodes, entropyThreshold]);

  const getNodeOpacity = useCallback((nodeId: string) => {
    if (!traversalResult) return 1;
    
    const isInvolved = traversalResult.visited_nodes.includes(nodeId) || 
                      traversalResult.boundary_nodes.includes(nodeId) ||
                      selectedNodes.includes(nodeId);
    
    return isInvolved ? 1 : 0.6;
  }, [traversalResult, selectedNodes]);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Network className="h-6 w-6 text-blue-300" />
          <h3 className="text-xl font-bold text-white">Knowledge Graph Visualization</h3>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleZoomIn}
            className="p-2 bg-blue-600/20 hover:bg-blue-600/40 rounded-lg transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="h-4 w-4 text-blue-300" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-2 bg-blue-600/20 hover:bg-blue-600/40 rounded-lg transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="h-4 w-4 text-blue-300" />
          </button>
          <button
            onClick={handleResetView}
            className="p-2 bg-blue-600/20 hover:bg-blue-600/40 rounded-lg transition-colors"
            title="Reset View"
          >
            <Eye className="h-4 w-4 text-blue-300" />
          </button>
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
          onMouseLeave={() => setTooltip(null)}
        >
          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
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
                onClick={(e) => handleNodeClick(node.id, e)}
                onMouseEnter={(e) => handleNodeHover(node.id, e)}
                onMouseLeave={() => setTooltip(null)}
              >
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={getNodeSize(node.id)}
                  fill={getNodeColor(node.id)}
                  stroke="#FFFFFF"
                  strokeWidth="2"
                  className="hover:stroke-4 transition-all duration-200"
                />
                <text
                  x={node.x}
                  y={node.y - getNodeSize(node.id) - 5}
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
          </g>
        </svg>

        {/* Tooltip */}
        {tooltip && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute z-10 bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg text-sm"
            style={{
              left: tooltip.x + 10,
              top: tooltip.y - 10,
              transform: 'translate(0, -100%)'
            }}
          >
            <div className="text-white font-medium">{tooltip.label}</div>
            <div className="text-gray-300">Status: {tooltip.status}</div>
            <div className="text-gray-300">Entropy: {tooltip.entropy.toFixed(3)}</div>
          </motion.div>
        )}
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
          <div className="text-blue-300 font-medium">Selected</div>
          <div className="text-white text-lg">{selectedNodes.length}</div>
        </div>
      </div>
    </div>
  );
};

export default GraphVisualizer;