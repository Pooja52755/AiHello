import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, BookOpen, Loader2, Zap } from 'lucide-react';
import toast from 'react-hot-toast';

interface TextInputProps {
  onTextProcess: (text: string) => void;
  loading: boolean;
}

const TextInput: React.FC<TextInputProps> = ({ onTextProcess, loading }) => {
  const [text, setText] = useState('');
  const [loadingGutenberg, setLoadingGutenberg] = useState(false);

  const sampleTexts = [
    {
      title: "War and Peace Sample",
      text: "War and Peace is a novel by Leo Tolstoy. The book chronicles the French invasion of Russia. Napoleon led his army into Russian territory. The Russian people fought bravely against the invaders. Pierre Bezukhov was one of the main characters. Natasha Rostova loved Prince Andrei. The war changed everyone's lives forever."
    },
    {
      title: "Scientific Text",
      text: "Machine learning algorithms process vast amounts of data. Neural networks mimic the human brain structure. Deep learning models require extensive training. Researchers develop new architectures continuously. Artificial intelligence transforms various industries. Scientists publish their findings regularly."
    }
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) {
      toast.error('Please enter some text to process');
      return;
    }
    onTextProcess(text);
    toast.success('Processing text...');
  };

  const loadGutenbergText = async () => {
    setLoadingGutenberg(true);
    try {
      const response = await fetch('http://localhost:5000/api/load-gutenberg', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ book_id: '2600' }),
      });

      if (!response.ok) {
        throw new Error('Failed to load text');
      }

      const data = await response.json();
      setText(data.text);
      toast.success('Loaded War and Peace text from Project Gutenberg');
    } catch (error) {
      console.error('Error loading text:', error);
      toast.error('Failed to load text from Project Gutenberg');
    } finally {
      setLoadingGutenberg(false);
    }
  };

  const loadSampleText = (sampleText: string) => {
    setText(sampleText);
    toast.success('Sample text loaded');
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 shadow-2xl">
      <div className="flex items-center space-x-3 mb-6">
        <FileText className="h-6 w-6 text-blue-300" />
        <h2 className="text-2xl font-bold text-white">Text Input & Processing</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="text" className="block text-sm font-medium text-blue-100 mb-2">
            Enter your text to build a knowledge graph:
          </label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-40 px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
            placeholder="Enter a paragraph or paste text here. The system will extract SVO triplets and build a knowledge graph..."
            disabled={loading}
          />
        </div>

        <div className="flex flex-wrap gap-3">
          <motion.button
            type="submit"
            disabled={loading || !text.trim()}
            className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-xl shadow-lg transition-all duration-200"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {loading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Zap className="h-5 w-5" />
            )}
            <span>{loading ? 'Processing...' : 'Build Knowledge Graph'}</span>
          </motion.button>

          <motion.button
            type="button"
            onClick={loadGutenbergText}
            disabled={loadingGutenberg}
            className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-xl shadow-lg transition-all duration-200"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {loadingGutenberg ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <BookOpen className="h-5 w-5" />
            )}
            <span>{loadingGutenberg ? 'Loading...' : 'Load War & Peace'}</span>
          </motion.button>
        </div>

        <div>
          <p className="text-sm font-medium text-blue-100 mb-3">Or try these sample texts:</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {sampleTexts.map((sample, index) => (
              <motion.button
                key={index}
                type="button"
                onClick={() => loadSampleText(sample.text)}
                className="p-4 bg-gray-800/30 hover:bg-gray-700/40 border border-gray-600 rounded-xl text-left transition-all duration-200"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <h4 className="font-medium text-white mb-1">{sample.title}</h4>
                <p className="text-sm text-gray-300 line-clamp-2">
                  {sample.text.substring(0, 100)}...
                </p>
              </motion.button>
            ))}
          </div>
        </div>
      </form>
    </div>
  );
};

export default TextInput;