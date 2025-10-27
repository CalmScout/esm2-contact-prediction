"""
Performance Analyzer for ESM2 Contact Prediction

This module provides specialized analysis functions for evaluating contact prediction
model performance, detecting overfitting, and generating training insights.

Key Features:
- Convergence analysis and detection
- Overfitting/underfitting assessment
- Training efficiency metrics
- Performance insights generation
- Stability and robustness analysis

Usage:
    from esm2_contact.analysis import PerformanceAnalyzer

    analyzer = PerformanceAnalyzer()
    insights = analyzer.analyze_performance(run_data)
    convergence = analyzer.analyze_convergence(training_history)
"""

import warnings
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class PerformanceAnalyzer:
    """
    Specialized performance analyzer for ESM2 contact prediction models.

    Provides analysis functions for training dynamics, model behavior,
    and performance optimization insights.

    Attributes:
        None (stateless analyzer)

    Example:
        analyzer = PerformanceAnalyzer()
        insights = analyzer.analyze_performance(run_data)
        convergence = analyzer.analyze_convergence(history_data)
    """

    def analyze_performance(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive performance analysis for a single run.

        Args:
            run_data (Dict[str, Any]): Run data from MLflow analyzer

        Returns:
            Dict[str, Any]: Performance analysis results
        """
        try:
            metrics = run_data.get('metrics', {})
            params = run_data.get('params', {})
            artifacts = run_data.get('artifacts', {})

            # Extract training history if available
            history = artifacts.get('training_history', {})
            if not history:
                # Try to create basic history from metrics
                history = self._create_history_from_metrics(metrics)

            analysis = {
                'final_performance': self._analyze_final_performance(metrics),
                'training_dynamics': self._analyze_training_dynamics(history),
                'convergence_analysis': self._analyze_convergence(history),
                'overfitting_assessment': self._assess_overfitting(history),
                'training_efficiency': self._calculate_training_efficiency(
                    run_data['run_info'], metrics, history
                ),
                'stability_analysis': self._analyze_stability(history),
                'parameter_impact': self._analyze_parameter_impact(params, metrics),
                'insights': []
            }

            # Generate insights
            analysis['insights'] = self._generate_performance_insights(analysis)

            return analysis

        except Exception as e:
            raise RuntimeError(f"Performance analysis failed: {e}")

    def _create_history_from_metrics(self, metrics: Dict[str, Any]) -> Dict[str, List[float]]:
        """Create minimal history data from available metrics."""
        history = {}

        # Create lists from scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and 'epoch' in key.lower():
                if key not in history:
                    history[key] = []
                history[key].append(value)

        return history

    def _analyze_final_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze final model performance."""
        performance = {}

        # Primary metrics
        if 'val_auc' in metrics:
            performance['val_auc'] = {
                'value': metrics['val_auc'],
                'rating': self._rate_auc(metrics['val_auc']),
                'interpretation': self._interpret_auc(metrics['val_auc'])
            }

        if 'val_precision_at_l' in metrics:
            performance['val_precision_at_l'] = {
                'value': metrics['val_precision_at_l'],
                'rating': self._rate_precision(metrics['val_precision_at_l']),
                'interpretation': self._interpret_precision(metrics['val_precision_at_l'])
            }

        if 'val_precision_at_l5' in metrics:
            performance['val_precision_at_l5'] = {
                'value': metrics['val_precision_at_l5'],
                'rating': self._rate_precision(metrics['val_precision_at_l5']),
                'interpretation': self._interpret_precision(metrics['val_precision_at_l5'])
            }

        return performance

    def _analyze_training_dynamics(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze training dynamics and patterns."""
        dynamics = {}

        if 'train_loss' in history and 'val_loss' in history:
            train_losses = np.array(history['train_loss'])
            val_losses = np.array(history['val_loss'])

            dynamics['loss_analysis'] = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': np.min(val_losses),
                'train_loss_reduction': train_losses[0] - train_losses[-1],
                'val_loss_reduction': val_losses[0] - val_losses[-1],
                'final_gap': train_losses[-1] - val_losses[-1],
                'min_gap': np.min(np.abs(train_losses - val_losses)),
                'overfitting_trend': self._analyze_overfitting_trend(train_losses, val_losses)
            }

        if 'val_auc' in history:
            val_aucs = np.array(history['val_auc'])
            dynamics['auc_analysis'] = {
                'final_auc': val_aucs[-1],
                'best_auc': np.max(val_aucs),
                'auc_improvement': val_aucs[-1] - val_aucs[0],
                'auc_stability': np.std(val_aucs[-min(10, len(val_aucs)):]),
                'convergence_epoch': self._find_convergence_epoch(val_aucs)
            }

        return dynamics

    def _analyze_convergence(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze convergence behavior."""
        convergence = {}

        if 'val_auc' in history:
            val_aucs = np.array(history['val_auc'])

            convergence.update({
                'converged': self._is_converged(val_aucs),
                'convergence_epoch': self._find_convergence_epoch(val_aucs),
                'final_stability': self._calculate_stability(val_aucs),
                'plateaued': self._is_plateaued(val_aucs),
                'improvement_rate': self._calculate_improvement_rate(val_aucs)
            })

        if 'train_loss' in history and 'val_loss' in history:
            train_losses = np.array(history['train_loss'])
            val_losses = np.array(history['val_loss'])

            convergence.update({
                'loss_convergence': self._is_converged(val_losses),
                'gap_stability': self._calculate_gap_stability(train_losses, val_losses)
            })

        return convergence

    def _assess_overfitting(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Assess overfitting/underfitting."""
        assessment = {}

        if 'train_loss' in history and 'val_loss' in history:
            train_losses = np.array(history['train_loss'])
            val_losses = np.array(history['val_loss'])

            # Calculate overfitting indicators
            final_gap = train_losses[-1] - val_losses[-1]
            max_gap = np.max(train_losses - val_losses)
            gap_trend = self._analyze_overfitting_trend(train_losses, val_losses)

            assessment['overfitting_score'] = self._calculate_overfitting_score(
                final_gap, max_gap, gap_trend
            )
            assessment['overfitting_status'] = self._classify_overfitting(
                assessment['overfitting_score']
            )
            assessment['recommendations'] = self._generate_overfitting_recommendations(
                assessment['overfitting_status']
            )

        return assessment

    def _calculate_training_efficiency(self, run_info: Dict[str, Any],
                                        metrics: Dict[str, Any],
                                        history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate training efficiency metrics."""
        efficiency = {}

        # Time-based efficiency
        if run_info.get('start_time') and run_info.get('end_time'):
            training_time = (run_info['end_time'] - run_info['start_time']).total_seconds()
            total_epochs = len(history.get('train_loss', []))

            efficiency['training_time'] = training_time
            efficiency['time_per_epoch'] = training_time / total_epochs if total_epochs > 0 else None

        # Performance efficiency
        if 'val_auc' in metrics and total_epochs > 0:
            efficiency['auc_per_epoch'] = metrics['val_auc'] / total_epochs
            efficiency['final_auc'] = metrics['val_auc']

        # Memory efficiency
        if 'gpu_memory' in metrics:
            efficiency['peak_gpu_memory'] = metrics['gpu_memory']
            efficiency['memory_efficiency'] = self._calculate_memory_efficiency(
                metrics.get('gpu_memory'), metrics
            )

        return efficiency

    def _analyze_stability(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze training stability."""
        stability = {}

        if 'val_auc' in history:
            val_aucs = np.array(history['val_auc'])

            stability.update({
                'final_stability': self._calculate_stability(val_aucs),
                'volatility': np.std(val_aucs),
                'trend': self._calculate_trend(val_aucs),
                'consistency': self._calculate_consistency(val_aucs)
            })

        return stability

    def _analyze_parameter_impact(self, params: Dict[str, Any],
                               metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of different hyperparameters."""
        impact = {}

        # Learning rate impact
        if 'learning_rate' in params and 'val_auc' in metrics:
            lr = params['learning_rate']
            auc = metrics['val_auc']
            impact['learning_rate'] = {
                'value': lr,
                'performance': auc,
                'optimal_range': self._suggest_optimal_lr(auc)
            }

        # Batch size impact
        if 'batch_size' in params and 'val_auc' in metrics:
            batch_size = params['batch_size']
            auc = metrics['val_auc']
            impact['batch_size'] = {
                'value': batch_size,
                'performance': auc,
                'efficiency_rating': self._rate_batch_efficiency(batch_size, auc)
            }

        # Model size impact
        model_size_params = ['base_channels', 'dropout_rate']
        for param in model_size_params:
            if param in params and 'val_auc' in metrics:
                impact[param] = {
                    'value': params[param],
                    'performance': metrics['val_auc']
                }

        return impact

    def _generate_performance_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable performance insights."""
        insights = []

        # Performance rating insights
        if 'final_performance' in analysis:
            perf = analysis['final_performance']

            if 'val_auc' in perf:
                auc_info = perf['val_auc']
                insights.append(
                    f"Final validation AUC: {auc_info['value']:.4f} ({auc_info['interpretation']})"
                )

            if 'val_precision_at_l' in perf:
                prec_info = perf['val_precision_at_l']
                insights.append(
                    f"Precision@L: {prec_info['value']:.4f} ({prec_info['interpretation']})"
                )

        # Training dynamics insights
        if 'training_dynamics' in analysis:
            dynamics = analysis['training_dynamics']

            if 'loss_analysis' in dynamics:
                loss_info = dynamics['loss_analysis']
                if loss_info['final_gap'] > 0.1:
                    insights.append(
                        f"⚠️ Large training/validation gap ({loss_info['final_gap']:.3f}) suggests overfitting"
                    )
                elif loss_info['final_gap'] < 0:
                    insights.append(
                        f"✅ Training loss lower than validation loss - possible underfitting"
                    )

            if 'auc_analysis' in dynamics:
                auc_info = dynamics['auc_analysis']
                if auc_info['converged']:
                    insights.append(
                        f"✅ Model converged in {auc_info['convergence_epoch']} epochs"
                    )
                else:
                    insights.append(
                        f"⚠️ Model did not converge within training period"
                    )

        # Convergence insights
        if 'convergence_analysis' in analysis:
            conv = analysis['convergence_analysis']

            if conv.get('converged'):
                insights.append(
                    f"✅ Stable convergence achieved with {conv['final_stability']:.3f} stability"
                )
            else:
                insights.append(
                    f"⚠️ Model did not achieve stable convergence"
                )

        # Overfitting insights
        if 'overfitting_assessment' in analysis:
            overfit = analysis['overfitting_assessment']

            if overfit['overfitting_status'] == 'severe':
                insights.append(
                    f"❌ Severe overfitting detected: {overfitting['overfitting_score']:.2f}"
                )
            elif overfit['overfitting_status'] == 'moderate':
                insights.append(
                    f"⚠️ Moderate overfitting detected: {overfitting['overfitting_score']:.2f}"
                )

            for rec in overfitting.get('recommendations', []):
                insights.append(f"💡 {rec}")

        return insights

    # Helper methods for analysis
    def _rate_auc(self, auc: float) -> str:
        """Rate AUC performance."""
        if auc >= 0.9:
            return "Excellent"
        elif auc >= 0.8:
            return "Very Good"
        elif auc >= 0.7:
            return "Good"
        elif auc >= 0.6:
            return "Moderate"
        else:
            return "Poor"

    def _interpret_auc(self, auc: float) -> str:
        """Interpret AUC value."""
        if auc >= 0.9:
            return "Excellent discrimination between contacting and non-contacting residues"
        elif auc >= 0.8:
            return "Strong discrimination ability"
        elif auc >= 0.7:
            return "Good discrimination ability"
        elif auc >= 0.6:
            return "Moderate discrimination ability"
        elif auc >= 0.5:
            return "Random performance"
        else:
            return "Poor discrimination ability"

    def _rate_precision(self, precision: float) -> str:
        """Rate precision performance."""
        if precision >= 0.8:
            return "Excellent"
        elif precision >= 0.6:
            return "Good"
        elif precision >= 0.4:
            return "Moderate"
        elif precision >= 0.2:
            return "Poor"
        else:
            return "Very Poor"

    def _interpret_precision(self, precision: float) -> str:
        """Interpret precision value."""
        if precision >= 0.8:
            return "High confidence in predictions"
        elif precision >= 0.6:
            return "Moderate confidence in predictions"
        elif precision >= 0.4:
            return "Low confidence in predictions"
        elif precision >= 0.2:
            return "Very low confidence in predictions"
        else:
            return "Random or poor performance"

    def _is_converged(self, values: np.ndarray, patience: int = 10, tolerance: float = 0.01) -> bool:
        """Check if series has converged."""
        if len(values) < patience * 2:
            return False

        # Check if last 'patience' values are within tolerance
        recent_values = values[-patience:]
        return np.std(recent_values) < tolerance

    def _find_convergence_epoch(self, values: np.ndarray) -> int:
        """Find epoch where convergence began."""
        if len(values) < 5:
            return 0

        best_value = np.max(values)
        convergence_threshold = best_value * 0.99

        for i in range(len(values) - 1, -1, -1):
            if values[i] >= convergence_threshold:
                return i + 1

        return len(values)

    def _calculate_stability(self, values: np.ndarray, window: int = 10) -> float:
        """Calculate stability of recent values."""
        if len(values) < window:
            return 0.0

        recent_values = values[-window:]
        return 1.0 / (1.0 + np.std(recent_values))

    def _calculate_improvement_rate(self, values: np.ndarray) -> float:
        """Calculate improvement rate."""
        if len(values) < 2:
            return 0.0

        return (values[-1] - values[0]) / values[0]

    def _is_plateaued(self, values: np.ndarray, window: int = 10, tolerance: float = 0.001) -> bool:
        """Check if series has plateaued."""
        if len(values) < window * 2:
            return False

        recent_values = values[-window:]
        return np.max(recent_values) - np.min(recent_values) < tolerance

    def _analyze_overfitting_trend(self, train_losses: np.ndarray, val_losses: np.ndarray) -> str:
        """Analyze overfitting trend."""
        if len(train_losses) < 10 or len(val_losses) < 10:
            return "insufficient_data"

        # Calculate gap trend
        gaps = train_losses - val_losses
        recent_gaps = gaps[-5:]  # Last 5 epochs
        gap_slope = np.polyfit(range(len(recent_gaps)), recent_gaps, 1)[0]

        if gap_slope > 0.01:
            return "increasing"
        elif gap_slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _calculate_overfitting_score(self, final_gap: float, max_gap: float, trend: str) -> float:
        """Calculate overfitting score."""
        base_score = min(final_gap / 0.5, 1.0)  # Normalize to [0, 1]

        # Adjust based on trend
        if trend == "increasing":
            return min(base_score * 1.2, 1.0)
        elif trend == "decreasing":
            return max(base_score * 0.8, 0.0)
        else:
            return base_score

    def _classify_overfitting(self, score: float) -> str:
        """Classify overfitting severity."""
        if score >= 0.8:
            return "severe"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.2:
            return "mild"
        else:
            return "minimal"

    def _generate_overfitting_recommendations(self, status: str) -> List[str]:
        """Generate overfitting recommendations."""
        recommendations = []

        if status == "severe":
            recommendations.extend([
                "Add dropout regularization (increase dropout_rate)",
                "Use stronger weight decay (increase weight_decay)",
                "Reduce model complexity (fewer layers/channels)",
                "Use early stopping with lower patience",
                "Increase training data or use data augmentation",
                "Reduce learning rate"
            ])
        elif status == "moderate":
            recommendations.extend([
                "Consider adding dropout if not used",
                "Slightly increase regularization strength",
                "Monitor validation performance closely"
            ])
        elif status == "mild":
            recommendations.extend([
                "Current regularization seems appropriate",
                "Monitor for changes during extended training"
            ])

        return recommendations

    def _calculate_memory_efficiency(self, memory_usage: float, metrics: Dict[str, Any]) -> str:
        """Calculate memory efficiency rating."""
        # This is a simplified efficiency calculation
        if memory_usage < 2.0:  # Less than 2GB
            return "excellent"
        elif memory_usage < 4.0:
            return "good"
        elif memory_usage < 8.0:
            return "moderate"
        else:
            return "poor"

    def _rate_batch_efficiency(self, batch_size: int, auc: float) -> str:
        """Rate batch size efficiency."""
        # This is a simplified rating
        if batch_size == 1:
            return "limited but memory-efficient"
        elif batch_size <= 4:
            return "good balance"
        elif batch_size <= 8:
            return "potentially efficient"
        else:
            return "may be inefficient"

    def _suggest_optimal_lr(self, auc: float) -> str:
        """Suggest optimal learning rate range based on performance."""
        if auc >= 0.85:
            return "1e-4 to 1e-3 (current training may be too fast)"
        elif auc >= 0.8:
            return "1e-3 to 1e-2 (optimal range)"
        elif auc >= 0.7:
            return "1e-3 to 5e-2 (may need learning rate adjustment)"
        else:
            return "1e-3 to 1e-1 (consider reducing learning rate)"

    def _calculate_gap_stability(self, train_losses: np.ndarray, val_losses: np.ndarray) -> float:
        """Calculate stability of training/validation gap."""
        gaps = train_losses - val_losses
        if len(gaps) < 5:
            return 1.0

        recent_gaps = gaps[-5:]
        return 1.0 / (1.0 + np.std(recent_gaps))

    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend in values."""
        if len(values) < 5:
            return "insufficient data"

        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "degrading"
        else:
            return "stable"

    def _calculate_consistency(self, values: np.ndarray) -> float:
        """Calculate consistency of values."""
        if len(values) < 3:
            return 1.0

        # Use coefficient of variation
        mean_val = np.mean(values)
        if mean_val == 0:
            return 1.0

        return 1.0 / (1.0 + np.std(values) / mean_val)