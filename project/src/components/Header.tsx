import React from 'react';
import { motion } from 'framer-motion';
import { Brain, GitBranch, Zap } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <motion.header 
      className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 shadow-2xl"
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <motion.div 
            className="flex items-center space-x-4"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <div className="relative">
              <Brain className="h-10 w-10 text-white" />
              <motion.div
                className="absolute -top-1 -right-1 h-4 w-4 bg-yellow-400 rounded-full"
                animate={{ 
                  scale: [1, 1.2, 1],
                  opacity: [0.7, 1, 0.7]
                }}
                transition={{ 
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white tracking-tight">
                KG Entropy Detector
              </h1>
              <p className="text-blue-100 text-sm font-medium">
                Sentence Boundary Detection via Knowledge Graph Traversal
              </p>
            </div>
          </motion.div>

          <div className="hidden md:flex items-center space-x-6">
            <motion.div 
              className="flex items-center space-x-2 text-blue-100"
              whileHover={{ scale: 1.1 }}
            >
              <GitBranch className="h-5 w-5" />
              <span className="text-sm font-medium">Graph Neural Network</span>
            </motion.div>
            <motion.div 
              className="flex items-center space-x-2 text-blue-100"
              whileHover={{ scale: 1.1 }}
            >
              <Zap className="h-5 w-5" />
              <span className="text-sm font-medium">BLT-Inspired Entropy</span>
            </motion.div>
          </div>
        </div>

        <motion.div 
          className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
        >
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="text-xl font-bold text-white">SVO Extraction</div>
            <div className="text-sm text-blue-100">Subject-Verb-Object Triplets</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="text-xl font-bold text-white">Entropy Analysis</div>
            <div className="text-sm text-blue-100">Boundary Detection Algorithm</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="text-xl font-bold text-white">Graph Traversal</div>
            <div className="text-sm text-blue-100">Intelligent Path Finding</div>
          </div>
        </motion.div>
      </div>
    </motion.header>
  );
};

export default Header;