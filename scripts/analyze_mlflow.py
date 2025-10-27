"""
Command-Line MLflow Analysis Tool for ESM2 Contact Prediction

This script provides comprehensive analysis capabilities for MLflow experiments,
including performance comparison, training dynamics visualization, and insights generation.

Usage:
    python scripts/analyze_mlflow.py --experiment-name esm2_contact_prediction --list-runs
    python scripts/analyze_mlflow.py --experiment-name esm2_contact_prediction --best-run --metric val_auc
    python scripts/analyze_mlflow.py --experiment-name esm2_contact_prediction --compare-runs --output comparison.html
    python scripts/analyze_mlflow.py --experiment-name esm2_contact_prediction --run-id <run_id> --detailed-analysis
    python scripts/analyze_mlflow.py --experiment-name esm2_contact_prediction --plot-training --output plots/
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from esm2_contact.analysis import MLflowAnalyzer, PerformanceAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze MLflow experiments for ESM2 Contact Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --experiment-name esm2_contact_prediction --list-runs
  %(prog)s --experiment-name esm2_contact_prediction --overview
  %(prog)s --experiment-name esm2_contact_prediction --best-run --metric val_auc
  %(prog)s --experiment-name esm2_contact_prediction --run-id abc123 --detailed-analysis
  %(prog)s --experiment-name esm2_contact_prediction --compare-runs --output comparison.csv
  %(prog)s --experiment-name esm2_contact_prediction --plot-training --output-dir plots/
        """
    )

    # Experiment selection
    parser.add_argument(
        "--experiment-name",
        default="esm2_contact_prediction",
        help="MLflow experiment name (default: esm2_contact_prediction)"
    )

    parser.add_argument(
        "--run-id",
        help="Specific run ID to analyze"
    )

    # Actions
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all available experiments"
    )

    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all runs in the experiment"
    )

    parser.add_argument(
        "--overview",
        action="store_true",
        help="Generate experiment overview with summary statistics"
    )

    parser.add_argument(
        "--best-run",
        action="store_true",
        help="Show the best performing run"
    )

    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="Show the most recent run"
    )

    parser.add_argument(
        "--compare-runs",
        action="store_true",
        help="Compare multiple runs in the experiment"
    )

    parser.add_argument(
        "--detailed-analysis",
        action="store_true",
        help="Perform detailed performance analysis (requires --run-id)"
    )

    parser.add_argument(
        "--plot-training",
        action="store_true",
        help="Generate training plots (loss curves, metrics over time)"
    )

    # Options
    parser.add_argument(
        "--metric",
        default="val_auc",
        help="Metric for optimization (default: val_auc)"
    )

    parser.add_argument(
        "--direction",
        choices=["max", "min"],
        default="max",
        help="Optimization direction (default: max)"
    )

    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI (default: file:///tmp/mlruns)"
    )

    parser.add_argument(
        "--output",
        help="Output file path (for CSV/HTML exports)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for plots and multiple files"
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv", "html", "excel"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top runs to show (default: 10)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def list_experiments(analyzer: MLflowAnalyzer, verbose: bool = False):
    """List all available experiments."""
    print("🔍 Available MLflow Experiments:")
    print("=" * 60)

    try:
        experiments = analyzer.list_experiments()

        if not experiments:
            print("❌ No experiments found")
            return

        for exp in experiments:
            print(f"📊 {exp['name']}")
            print(f"   ID: {exp['experiment_id']}")
            print(f"   Created: {exp['creation_time']}")
            print(f"   Last Updated: {exp['last_update']}")
            if exp.get('tags'):
                print(f"   Tags: {exp['tags']}")
            print()

        print(f"✅ Found {len(experiments)} experiments")

    except Exception as e:
        print(f"❌ Failed to list experiments: {e}")


def list_runs(analyzer: MLflowAnalyzer, experiment_name: str,
              top_k: int = 10, verbose: bool = False):
    """List runs in an experiment."""
    print(f"🔍 Runs in experiment: {experiment_name}")
    print("=" * 70)

    try:
        experiment_data = analyzer.load_experiment(experiment_name, include_artifacts=False)
        runs = experiment_data['runs']

        if not runs:
            print("❌ No runs found in experiment")
            return

        # Sort by validation AUC if available
        sorted_runs = sorted(
            runs.items(),
            key=lambda x: x[1].get('metrics', {}).get('val_auc', 0),
            reverse=True
        )

        print(f"{'Run ID':<20} {'Status':<10} {'Val AUC':<8} {'Val Loss':<8} {'Epochs':<8} {'Run Name'}")
        print("-" * 90)

        for i, (run_id, run_data) in enumerate(sorted_runs[:top_k]):
            run_info = run_data.get('run_info', {})
            metrics = run_data.get('metrics', {})

            status = run_info.get('status', 'UNKNOWN')
            val_auc = metrics.get('val_auc', 0)
            val_loss = metrics.get('val_loss', 0)
            epochs = metrics.get('total_epochs', 0)
            run_name = run_info.get('run_name', 'Unknown')[:30]

            print(f"{run_id[:18]:<20} {status:<10} {val_auc:<8.4f} {val_loss:<8.4f} {epochs:<8} {run_name}")

        if len(runs) > top_k:
            print(f"... and {len(runs) - top_k} more runs")

        print(f"\n✅ Total: {len(runs)} runs")

    except Exception as e:
        print(f"❌ Failed to list runs: {e}")


def show_experiment_overview(analyzer: MLflowAnalyzer, experiment_name: str):
    """Generate experiment overview."""
    print(f"📊 Experiment Overview: {experiment_name}")
    print("=" * 70)

    try:
        experiment_data = analyzer.load_experiment(experiment_name, include_artifacts=False)
        overview = analyzer.analyze_experiment_overview(experiment_data)

        # Basic statistics
        print(f"📈 Basic Statistics:")
        print(f"   Total runs: {overview['total_runs']}")
        print(f"   Completed: {overview['completed_runs']}")
        print(f"   Failed: {overview['failed_runs']}")
        print(f"   Success rate: {overview['success_rate']:.1%}")
        print()

        # Performance statistics
        print(f"🎯 Performance Metrics:")
        for metric_name, stats in overview['performance'].items():
            if stats['count'] > 0:
                print(f"   {metric_name}:")
                print(f"     Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"     Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"     Count: {stats['count']}")
        print()

        # Best run
        best_run = analyzer.get_best_run(experiment_data, metric="val_auc")
        if best_run:
            run_info = best_run.get('run_info', {})
            metrics = best_run.get('metrics', {})
            print(f"🏆 Best Run:")
            print(f"   Run ID: {run_info.get('run_id', 'Unknown')}")
            print(f"   Run Name: {run_info.get('run_name', 'Unknown')}")
            print(f"   Val AUC: {metrics.get('val_auc', 0):.4f}")
            print(f"   Val Precision@L: {metrics.get('val_precision_at_l', 0):.4f}")
            print(f"   Val Precision@L5: {metrics.get('val_precision_at_l5', 0):.4f}")

        print(f"\n✅ Overview generated successfully")

    except Exception as e:
        print(f"❌ Failed to generate overview: {e}")


def show_best_run(analyzer: MLflowAnalyzer, experiment_name: str,
                  metric: str = "val_auc", direction: str = "max"):
    """Show the best performing run."""
    print(f"🏆 Best Run (by {metric}, {direction})")
    print("=" * 50)

    try:
        experiment_data = analyzer.load_experiment(experiment_name, include_artifacts=True)
        best_run = analyzer.get_best_run(experiment_data, metric=metric, direction=direction)

        if not best_run:
            print(f"❌ No runs found with metric '{metric}'")
            return

        # Run information
        run_info = best_run.get('run_info', {})
        metrics = best_run.get('metrics', {})
        params = best_run.get('params', {})

        print(f"🎯 Run Information:")
        print(f"   Run ID: {run_info.get('run_id', 'Unknown')}")
        print(f"   Run Name: {run_info.get('run_name', 'Unknown')}")
        print(f"   Status: {run_info.get('status', 'Unknown')}")
        print(f"   Start Time: {run_info.get('start_time', 'Unknown')}")
        print(f"   End Time: {run_info.get('end_time', 'Unknown')}")
        print()

        print(f"📊 Performance Metrics:")
        for metric_name, value in sorted(metrics.items()):
            print(f"   {metric_name}: {value:.6f}" if isinstance(value, float) else f"   {metric_name}: {value}")
        print()

        print(f"⚙️ Hyperparameters:")
        for param_name, value in sorted(params.items()):
            print(f"   {param_name}: {value}")

        print(f"\n✅ Best run analysis completed")

    except Exception as e:
        print(f"❌ Failed to analyze best run: {e}")


def show_latest_run(analyzer: MLflowAnalyzer, experiment_name: str):
    """Show the most recent run."""
    print(f"🕒 Latest Run in Experiment")
    print("=" * 40)

    try:
        latest_run = analyzer.get_latest_run(experiment_name)

        if not latest_run:
            print(f"❌ No runs found for experiment: {experiment_name}")
            return

        # Run information
        run_info = latest_run.get('run_info', {})
        metrics = latest_run.get('metrics', {})
        params = latest_run.get('params', {})

        print(f"🎯 Run Information:")
        print(f"   Run ID: {run_info.get('run_id', 'Unknown')}")
        print(f"   Run Name: {run_info.get('run_name', 'Unknown')}")
        print(f"   Status: {run_info.get('status', 'Unknown')}")
        print(f"   Start Time: {run_info.get('start_time', 'Unknown')}")
        print()

        print(f"📊 Performance Metrics:")
        for metric_name, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"   {metric_name}: {value:.6f}")
            else:
                print(f"   {metric_name}: {value}")

        print(f"\n✅ Latest run analysis completed")

    except Exception as e:
        print(f"❌ Failed to analyze latest run: {e}")


def compare_runs(analyzer: MLflowAnalyzer, experiment_name: str,
                output_path: Optional[str] = None, format: str = "json"):
    """Compare multiple runs in an experiment."""
    print(f"📊 Run Comparison for Experiment: {experiment_name}")
    print("=" * 60)

    try:
        experiment_data = analyzer.load_experiment(experiment_name, include_artifacts=False)
        run_list = list(experiment_data['runs'].values())

        if not run_list:
            print("❌ No runs found to compare")
            return

        # Create comparison table
        comparison_df = analyzer.compare_runs(run_list)

        print(f"📋 Comparison Table (Top 10 runs):")
        print(comparison_df.head(10).to_string(index=False))

        # Export if requested
        if output_path:
            analyzer.export_experiment_data(experiment_data, output_path, format)
            print(f"\n💾 Comparison exported to: {output_path}")

        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"   Total runs compared: {len(comparison_df)}")

        if 'val_auc' in comparison_df.columns:
            print(f"   Val AUC - Mean: {comparison_df['val_auc'].mean():.4f}")
            print(f"   Val AUC - Std: {comparison_df['val_auc'].std():.4f}")
            print(f"   Val AUC - Range: [{comparison_df['val_auc'].min():.4f}, {comparison_df['val_auc'].max():.4f}]")

        print(f"\n✅ Run comparison completed")

    except Exception as e:
        print(f"❌ Failed to compare runs: {e}")


def detailed_analysis(analyzer: MLflowAnalyzer, performance_analyzer: PerformanceAnalyzer,
                     experiment_name: str, run_id: str, output_dir: Optional[str] = None):
    """Perform detailed performance analysis of a specific run."""
    print(f"🔬 Detailed Performance Analysis")
    print("=" * 50)
    print(f"Experiment: {experiment_name}")
    print(f"Run ID: {run_id}")
    print()

    try:
        # Load run data
        run_data = analyzer.get_run(experiment_name, run_id)

        if not run_data:
            print(f"❌ Run not found: {run_id}")
            return

        # Perform detailed analysis
        analysis_results = performance_analyzer.analyze_performance(run_data)

        # Display results
        print(f"🎯 Final Performance Assessment:")
        final_perf = analysis_results['final_performance']
        for metric, assessment in final_perf.items():
            print(f"   {metric}: {assessment['value']:.4f} - {assessment['assessment']}")
            print(f"      Percentile: {assessment['percentile']:.1f}%")
        print()

        print(f"📈 Training Dynamics:")
        dynamics = analysis_results['training_dynamics']
        print(f"   Learning Progress: {dynamics['learning_progress']['assessment']}")
        print(f"   Overfitting Status: {dynamics['overfitting_analysis']['status']}")
        if dynamics['overfitting_analysis']['gap_analysis']:
            gap = dynamics['overfitting_analysis']['gap_analysis']
            print(f"   Val-Train Gap: {gap['final_gap']:.4f} (trend: {gap['trend']})")
        print()

        print(f"⚡ Convergence Analysis:")
        convergence = analysis_results['convergence_analysis']
        if convergence['convergence_detected']:
            print(f"   Convergence: ✅ Detected at epoch {convergence['convergence_epoch']}")
            print(f"   Convergence Rate: {convergence['convergence_rate']:.4f}/epoch")
        else:
            print(f"   Convergence: ❌ Not detected")
        print()

        print(f"🎪 Training Efficiency:")
        efficiency = analysis_results['training_efficiency']
        print(f"   Total Time: {efficiency['total_time_minutes']:.1f} minutes")
        print(f"   Time per Epoch: {efficiency['time_per_epoch']:.1f} seconds")
        print(f"   Training Speed: {efficiency['samples_per_second']:.1f} samples/second")
        print()

        print(f"🔒 Stability Analysis:")
        stability = analysis_results['stability_analysis']
        print(f"   Loss Stability: {stability['loss_stability']['assessment']}")
        print(f"   Performance Consistency: {stability['performance_consistency']['assessment']}")
        if stability['gradient_analysis']:
            grad_analysis = stability['gradient_analysis']
            print(f"   Gradient Norm: {grad_analysis['final_norm']:.4f} (trend: {grad_analysis['trend']})")

        # Generate plots if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"\n📊 Generating analysis plots...")
            performance_analyzer.plot_performance_analysis(
                run_data,
                output_dir=str(output_path),
                show_plots=False
            )
            print(f"✅ Plots saved to: {output_path}")

        # Export analysis results
        if output_dir:
            import json
            results_path = output_path / "detailed_analysis.json"
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"✅ Analysis results exported to: {results_path}")

        print(f"\n✅ Detailed analysis completed")

    except Exception as e:
        print(f"❌ Failed to perform detailed analysis: {e}")


def plot_training_curves(analyzer: MLflowAnalyzer, performance_analyzer: PerformanceAnalyzer,
                        experiment_name: str, run_id: Optional[str] = None,
                        output_dir: Optional[str] = None):
    """Generate training curve plots."""
    print(f"📈 Training Curve Visualization")
    print("=" * 40)

    try:
        if run_id:
            # Plot specific run
            print(f"📊 Plotting training curves for run: {run_id}")
            run_data = analyzer.get_run(experiment_name, run_id)

            if not run_data:
                print(f"❌ Run not found: {run_id}")
                return

            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                performance_analyzer.plot_performance_analysis(
                    run_data,
                    output_dir=str(output_path),
                    show_plots=False
                )
                print(f"✅ Training curves saved to: {output_path}")
            else:
                performance_analyzer.plot_performance_analysis(
                    run_data,
                    show_plots=True
                )
        else:
            # Plot best runs
            print(f"📊 Plotting training curves for top runs in experiment")
            experiment_data = analyzer.load_experiment(experiment_name, include_artifacts=True)

            # Get top 3 runs by validation AUC
            runs = experiment_data['runs']
            sorted_runs = sorted(
                runs.items(),
                key=lambda x: x[1].get('metrics', {}).get('val_auc', 0),
                reverse=True
            )[:3]

            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

            for i, (rid, run_data) in enumerate(sorted_runs):
                print(f"   Plotting run {i+1}/3: {rid[:18]}...")

                if output_dir:
                    run_output_dir = output_path / f"run_{rid[:8]}"
                    run_output_dir.mkdir(exist_ok=True)

                    performance_analyzer.plot_performance_analysis(
                        run_data,
                        output_dir=str(run_output_dir),
                        show_plots=False
                    )
                else:
                    performance_analyzer.plot_performance_analysis(
                        run_data,
                        show_plots=False
                    )

            if output_dir:
                print(f"✅ Training curves saved to: {output_path}")
            else:
                print("✅ Training curves generated (plots displayed)")

        print(f"\n✅ Training curve visualization completed")

    except Exception as e:
        print(f"❌ Failed to generate training curves: {e}")


def main():
    """Main analysis function."""
    parser = setup_parser()
    args = parser.parse_args()

    # Initialize analyzers
    try:
        analyzer = MLflowAnalyzer(tracking_uri=args.tracking_uri)
        performance_analyzer = PerformanceAnalyzer()

        if args.verbose:
            print(f"🔧 Initialized MLflow analyzer")
            if args.tracking_uri:
                print(f"   Tracking URI: {args.tracking_uri}")

    except Exception as e:
        print(f"❌ Failed to initialize analyzers: {e}")
        sys.exit(1)

    # Execute requested actions
    if args.list_experiments:
        list_experiments(analyzer, args.verbose)

    elif args.list_runs:
        list_runs(analyzer, args.experiment_name, args.top_k, args.verbose)

    elif args.overview:
        show_experiment_overview(analyzer, args.experiment_name)

    elif args.best_run:
        show_best_run(analyzer, args.experiment_name, args.metric, args.direction)

    elif args.latest_run:
        show_latest_run(analyzer, args.experiment_name)

    elif args.compare_runs:
        compare_runs(analyzer, args.experiment_name, args.output, args.format)

    elif args.detailed_analysis:
        if not args.run_id:
            print("❌ --detailed-analysis requires --run-id")
            sys.exit(1)
        detailed_analysis(
            analyzer, performance_analyzer,
            args.experiment_name, args.run_id,
            args.output_dir
        )

    elif args.plot_training:
        plot_training_curves(
            analyzer, performance_analyzer,
            args.experiment_name, args.run_id,
            args.output_dir
        )

    else:
        # Default action: show overview
        show_experiment_overview(analyzer, args.experiment_name)


if __name__ == "__main__":
    main()