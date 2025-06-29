"""
Comprehensive Evaluation System for Knowledge Graph Sentence Boundary Detection

This module provides extensive evaluation capabilities including:
- Standard classification metrics
- Boundary-specific metrics
- Traversal efficiency evaluation
- Cross-validation and statistical testing
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import logging
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, 
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BoundaryEvaluationMetrics:
    """
    Comprehensive evaluation metrics for sentence boundary detection
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_boundary_detection(self, 
                                  predicted_boundaries: List[Dict[str, Any]],
                                  ground_truth_sentences: Dict[str, int]) -> Dict[str, float]:
        """
        Evaluate boundary detection results against ground truth
        
        Args:
            predicted_boundaries: List of boundary detection results
            ground_truth_sentences: Mapping of nodes to sentence IDs
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        metrics = {}
        
        # Standard classification metrics
        classification_metrics = self._evaluate_classification(
            predicted_boundaries, ground_truth_sentences
        )
        metrics.update(classification_metrics)
        
        # Boundary-specific metrics
        boundary_metrics = self._evaluate_boundary_quality(
            predicted_boundaries, ground_truth_sentences
        )
        metrics.update(boundary_metrics)
        
        # Traversal efficiency metrics
        efficiency_metrics = self._evaluate_traversal_efficiency(
            predicted_boundaries, ground_truth_sentences
        )
        metrics.update(efficiency_metrics)
        
        # Sentence grouping metrics
        grouping_metrics = self._evaluate_sentence_grouping(
            predicted_boundaries, ground_truth_sentences
        )
        metrics.update(grouping_metrics)
        
        return metrics
    
    def _evaluate_classification(self, 
                                predicted_boundaries: List[Dict[str, Any]],
                                ground_truth_sentences: Dict[str, int]) -> Dict[str, float]:
        """Evaluate as binary classification problem"""
        
        y_true = []
        y_pred = []
        y_scores = []
        
        for result in predicted_boundaries:
            start_node = result['start_node']
            predicted_nodes = set(result['sentence_nodes'])
            
            if start_node not in ground_truth_sentences:
                continue
            
            true_sentence_id = ground_truth_sentences[start_node]
            
            # For each predicted node, determine if it's correctly classified
            for node in predicted_nodes:
                if node in ground_truth_sentences:
                    true_same_sentence = ground_truth_sentences[node] == true_sentence_id
                    pred_same_sentence = True  # All predicted nodes are in same sentence
                    
                    y_true.append(1 if true_same_sentence else 0)
                    y_pred.append(1 if pred_same_sentence else 0)
                    
                    # Use entropy score as confidence
                    entropy_score = result['entropy_scores'].get(node, 0.5)
                    y_scores.append(1 - entropy_score)  # Lower entropy = higher confidence
        
        if not y_true:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        # AUC if we have probability scores
        auc = 0.0
        if y_scores and len(set(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc = 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'auc': float(auc)
        }
    
    def _evaluate_boundary_quality(self, 
                                  predicted_boundaries: List[Dict[str, Any]],
                                  ground_truth_sentences: Dict[str, int]) -> Dict[str, float]:
        """Evaluate quality of boundary detection"""
        
        boundary_precisions = []
        boundary_recalls = []
        early_stop_accuracy = []
        
        for result in predicted_boundaries:
            start_node = result['start_node']
            predicted_nodes = result['sentence_nodes']
            boundary_detected = result.get('boundary_detected', False)
            boundary_node = result.get('boundary_node', None)
            
            if start_node not in ground_truth_sentences:
                continue
            
            true_sentence_id = ground_truth_sentences[start_node]
            true_sentence_nodes = {
                node for node, sent_id in ground_truth_sentences.items()
                if sent_id == true_sentence_id
            }
            
            # Boundary precision: How accurate is the stopping point?
            if boundary_detected and boundary_node:
                # Check if boundary was detected at correct position
                if boundary_node in ground_truth_sentences:
                    boundary_sentence_id = ground_truth_sentences[boundary_node]
                    correct_boundary = boundary_sentence_id != true_sentence_id
                    boundary_precisions.append(1.0 if correct_boundary else 0.0)
                else:
                    boundary_precisions.append(0.0)
            
            # Boundary recall: Did we stop when we should have?
            predicted_set = set(predicted_nodes)
            should_have_stopped = len(predicted_set - true_sentence_nodes) > 0
            did_stop = boundary_detected
            
            if should_have_stopped:
                boundary_recalls.append(1.0 if did_stop else 0.0)
            elif not should_have_stopped and not did_stop:
                boundary_recalls.append(1.0)  # Correctly didn't stop
            
            # Early stopping accuracy
            ideal_stop_point = len(true_sentence_nodes)
            actual_stop_point = len(predicted_nodes)
            
            if ideal_stop_point > 0:
                stop_accuracy = 1.0 - abs(actual_stop_point - ideal_stop_point) / ideal_stop_point
                early_stop_accuracy.append(max(0.0, stop_accuracy))
        
        return {
            'boundary_precision': np.mean(boundary_precisions) if boundary_precisions else 0.0,
            'boundary_recall': np.mean(boundary_recalls) if boundary_recalls else 0.0,
            'early_stop_accuracy': np.mean(early_stop_accuracy) if early_stop_accuracy else 0.0
        }
    
    def _evaluate_traversal_efficiency(self, 
                                     predicted_boundaries: List[Dict[str, Any]],
                                     ground_truth_sentences: Dict[str, int]) -> Dict[str, float]:
        """Evaluate efficiency of traversal algorithm"""
        
        efficiency_scores = []
        redundancy_scores = []
        path_optimality = []
        
        for result in predicted_boundaries:
            start_node = result['start_node']
            traversal_path = result.get('traversal_path', [])
            predicted_nodes = result['sentence_nodes']
            
            if start_node not in ground_truth_sentences:
                continue
            
            true_sentence_id = ground_truth_sentences[start_node]
            true_sentence_nodes = {
                node for node, sent_id in ground_truth_sentences.items()
                if sent_id == true_sentence_id
            }
            
            # Efficiency: ratio of useful nodes to total visited
            useful_nodes = set(predicted_nodes).intersection(true_sentence_nodes)
            total_visited = len(predicted_nodes)
            
            if total_visited > 0:
                efficiency = len(useful_nodes) / total_visited
                efficiency_scores.append(efficiency)
            
            # Redundancy: how many unnecessary nodes were visited
            unnecessary_nodes = set(predicted_nodes) - true_sentence_nodes
            redundancy = len(unnecessary_nodes) / max(1, len(true_sentence_nodes))
            redundancy_scores.append(redundancy)
            
            # Path optimality: compare to shortest path
            if len(traversal_path) > 1:
                path_length = len(traversal_path)
                optimal_length = len(true_sentence_nodes)
                optimality = optimal_length / max(1, path_length)
                path_optimality.append(min(1.0, optimality))
        
        return {
            'traversal_efficiency': np.mean(efficiency_scores) if efficiency_scores else 0.0,
            'traversal_redundancy': np.mean(redundancy_scores) if redundancy_scores else 0.0,
            'path_optimality': np.mean(path_optimality) if path_optimality else 0.0
        }
    
    def _evaluate_sentence_grouping(self, 
                                   predicted_boundaries: List[Dict[str, Any]],
                                   ground_truth_sentences: Dict[str, int]) -> Dict[str, float]:
        """Evaluate sentence-level grouping performance"""
        
        sentence_f1_scores = []
        sentence_precision_scores = []
        sentence_recall_scores = []
        
        # Group predictions by sentence
        sentence_predictions = defaultdict(list)
        for result in predicted_boundaries:
            start_node = result['start_node']
            if start_node in ground_truth_sentences:
                true_sentence_id = ground_truth_sentences[start_node]
                sentence_predictions[true_sentence_id].append(result)
        
        # Evaluate each sentence
        for sentence_id, predictions in sentence_predictions.items():
            true_nodes = {
                node for node, sent_id in ground_truth_sentences.items()
                if sent_id == sentence_id
            }
            
            if not true_nodes:
                continue
            
            # Combine all predictions for this sentence
            all_predicted_nodes = set()
            for pred in predictions:
                all_predicted_nodes.update(pred['sentence_nodes'])
            
            # Calculate sentence-level metrics
            intersection = all_predicted_nodes.intersection(true_nodes)
            
            precision = len(intersection) / len(all_predicted_nodes) if all_predicted_nodes else 0.0
            recall = len(intersection) / len(true_nodes) if true_nodes else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            sentence_precision_scores.append(precision)
            sentence_recall_scores.append(recall)
            sentence_f1_scores.append(f1)
        
        return {
            'sentence_precision': np.mean(sentence_precision_scores) if sentence_precision_scores else 0.0,
            'sentence_recall': np.mean(sentence_recall_scores) if sentence_recall_scores else 0.0,
            'sentence_f1': np.mean(sentence_f1_scores) if sentence_f1_scores else 0.0
        }

class CrossValidationEvaluator:
    """
    Cross-validation evaluation for robust model assessment
    """
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.fold_results = []
        
    def evaluate_with_cv(self, 
                        trainer,
                        texts: List[str],
                        random_state: int = 42) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation
        """
        
        np.random.seed(random_state)
        
        # Shuffle texts
        shuffled_texts = texts.copy()
        np.random.shuffle(shuffled_texts)
        
        # Split into folds
        fold_size = len(shuffled_texts) // self.n_folds
        folds = []
        
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_folds - 1 else len(shuffled_texts)
            folds.append(shuffled_texts[start_idx:end_idx])
        
        fold_metrics = []
        
        for fold_idx in range(self.n_folds):
            logger.info(f"Evaluating fold {fold_idx + 1}/{self.n_folds}")
            
            # Prepare train/test split
            test_texts = folds[fold_idx]
            train_texts = []
            for i, fold in enumerate(folds):
                if i != fold_idx:
                    train_texts.extend(fold)
            
            # Train on fold
            train_loader, val_loader = trainer.prepare_training_data(
                train_texts, validation_split=0.1
            )
            
            # Quick training for CV (fewer epochs)
            history = trainer.train(train_loader, val_loader, epochs=20, save_best=False)
            
            # Evaluate on test fold
            test_loader, _ = trainer.prepare_training_data(
                test_texts, validation_split=0.0
            )
            
            metrics = trainer.evaluate_model(test_loader)
            fold_metrics.append(metrics)
            
            logger.info(f"Fold {fold_idx + 1} - F1: {metrics['f1_score']:.4f}, "
                       f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Aggregate results
        aggregated_metrics = self._aggregate_fold_metrics(fold_metrics)
        
        return {
            'cv_metrics': aggregated_metrics,
            'fold_results': fold_metrics,
            'n_folds': self.n_folds
        }
    
    def _aggregate_fold_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across folds"""
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        aggregated = {}
        
        for metric in metric_names:
            values = [fold[metric] for fold in fold_metrics if metric in fold]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        return aggregated

class VisualizationTools:
    """
    Tools for visualizing evaluation results
    """
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], 
                             save_path: Optional[str] = None):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plots
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        
        # Accuracy plots
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
        axes[1, 0].plot(loss_diff)
        axes[1, 0].set_title('Train-Val Loss Difference')
        
        # Accuracy difference
        acc_diff = [abs(t - v) for t, v in zip(history['train_acc'], history['val_acc'])]
        axes[1, 1].plot(acc_diff)
        axes[1, 1].set_title('Train-Val Accuracy Difference')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray,
                             class_names: List[str] = None,
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""
        
        if class_names is None:
            class_names = ['Same Sentence', 'Boundary']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, float],
                               save_path: Optional[str] = None):
        """Plot metrics comparison"""
        
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class EvaluationReporter:
    """
    Generate comprehensive evaluation reports
    """
    
    def __init__(self):
        self.report_data = {}
    
    def generate_report(self, 
                       evaluation_results: Dict[str, Any],
                       model_config: Dict[str, Any],
                       training_history: Optional[Dict[str, List[float]]] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("# Knowledge Graph Sentence Boundary Detection - Evaluation Report\n")
        
        # Model configuration
        report.append("## Model Configuration")
        for key, value in model_config.items():
            report.append(f"- **{key}**: {value}")
        report.append("")
        
        # Training summary
        if training_history:
            report.append("## Training Summary")
            final_train_loss = training_history['train_loss'][-1]
            final_val_loss = training_history['val_loss'][-1]
            final_train_acc = training_history['train_acc'][-1]
            final_val_acc = training_history['val_acc'][-1]
            
            report.append(f"- **Final Training Loss**: {final_train_loss:.4f}")
            report.append(f"- **Final Validation Loss**: {final_val_loss:.4f}")
            report.append(f"- **Final Training Accuracy**: {final_train_acc:.4f}")
            report.append(f"- **Final Validation Accuracy**: {final_val_acc:.4f}")
            report.append("")
        
        # Classification metrics
        if 'classification_report' in evaluation_results:
            report.append("## Classification Metrics")
            class_report = evaluation_results['classification_report']
            
            report.append(f"- **Overall Accuracy**: {class_report['accuracy']:.4f}")
            report.append(f"- **Macro Precision**: {class_report['macro avg']['precision']:.4f}")
            report.append(f"- **Macro Recall**: {class_report['macro avg']['recall']:.4f}")
            report.append(f"- **Macro F1-Score**: {class_report['macro avg']['f1-score']:.4f}")
            report.append("")
            
            # Per-class metrics
            report.append("### Per-Class Performance")
            for class_name in ['Same Sentence', 'Boundary']:
                if class_name in class_report:
                    class_metrics = class_report[class_name]
                    report.append(f"**{class_name}:**")
                    report.append(f"- Precision: {class_metrics['precision']:.4f}")
                    report.append(f"- Recall: {class_metrics['recall']:.4f}")
                    report.append(f"- F1-Score: {class_metrics['f1-score']:.4f}")
                    report.append("")
        
        # Boundary-specific metrics
        boundary_metrics = [
            'boundary_precision', 'boundary_recall', 'early_stop_accuracy',
            'traversal_efficiency', 'traversal_redundancy', 'path_optimality',
            'sentence_precision', 'sentence_recall', 'sentence_f1'
        ]
        
        boundary_results = {k: v for k, v in evaluation_results.items() if k in boundary_metrics}
        if boundary_results:
            report.append("## Boundary Detection Metrics")
            for metric, value in boundary_results.items():
                formatted_name = metric.replace('_', ' ').title()
                report.append(f"- **{formatted_name}**: {value:.4f}")
            report.append("")
        
        # Cross-validation results
        if 'cv_metrics' in evaluation_results:
            report.append("## Cross-Validation Results")
            cv_metrics = evaluation_results['cv_metrics']
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if f'{metric}_mean' in cv_metrics:
                    mean_val = cv_metrics[f'{metric}_mean']
                    std_val = cv_metrics[f'{metric}_std']
                    min_val = cv_metrics[f'{metric}_min']
                    max_val = cv_metrics[f'{metric}_max']
                    
                    formatted_name = metric.replace('_', ' ').title()
                    report.append(f"**{formatted_name}:**")
                    report.append(f"- Mean: {mean_val:.4f} ± {std_val:.4f}")
                    report.append(f"- Range: [{min_val:.4f}, {max_val:.4f}]")
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        recommendations = self._generate_recommendations(evaluation_results)
        for rec in recommendations:
            report.append(f"- {rec}")
        
        return "\n".join(report)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Check accuracy
        accuracy = results.get('accuracy', 0.0)
        if accuracy < 0.7:
            recommendations.append("Consider increasing model complexity or training time (accuracy < 0.7)")
        elif accuracy > 0.95:
            recommendations.append("Check for overfitting (very high accuracy)")
        
        # Check boundary precision
        boundary_precision = results.get('boundary_precision', 0.0)
        if boundary_precision < 0.6:
            recommendations.append("Improve boundary detection by tuning entropy thresholds")
        
        # Check traversal efficiency
        efficiency = results.get('traversal_efficiency', 0.0)
        if efficiency < 0.8:
            recommendations.append("Optimize traversal algorithm to reduce unnecessary node visits")
        
        # Check F1 score
        f1_score = results.get('f1_score', 0.0)
        if f1_score < 0.6:
            recommendations.append("Balance precision and recall - consider class weighting or threshold tuning")
        
        if not recommendations:
            recommendations.append("Model performance looks good! Consider testing on more diverse datasets.")
        
        return recommendations
    
    def save_report(self, 
                   report_content: str, 
                   filepath: str,
                   results_dict: Optional[Dict[str, Any]] = None) -> None:
        """Save report to file"""
        
        # Save markdown report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save JSON results if provided
        if results_dict:
            json_path = Path(filepath).with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {filepath}")
