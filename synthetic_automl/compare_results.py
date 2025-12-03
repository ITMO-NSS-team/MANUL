
import os
from typing import Dict, Any, List
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compare_all_results(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Сравнивает результаты всех методов.
    
    Args:
        all_results: словарь {method_name: result_dict}
    
    Returns:
        Dict со сравнительными метриками
    """
    comparison = {
        'methods': [],
        'test_mse': [],
        'test_mae': [],
        'test_r2': [],
        'training_time': [],
        'best_epoch': []
    }
    
    for method, result in all_results.items():
        if 'error' in result:
            continue
        
        comparison['methods'].append(method)
        comparison['test_mse'].append(result['test_mse'])
        comparison['test_mae'].append(result['test_mae'])
        comparison['test_r2'].append(result['test_r2'])
        comparison['training_time'].append(result['training_time_seconds'])
        comparison['best_epoch'].append(result['best_epoch'])

    if 'baseline' in all_results and 'error' not in all_results['baseline']:
        baseline_mse = all_results['baseline']['test_mse']
        comparison['improvement_over_baseline'] = {}
        
        for method, result in all_results.items():
            if 'error' not in result and method != 'baseline':
                improvement = (baseline_mse - result['test_mse']) / baseline_mse * 100
                comparison['improvement_over_baseline'][method] = improvement
    

    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        best_method = min(valid_results, key=lambda x: valid_results[x]['test_mse'])
        comparison['best_method'] = best_method
        comparison['best_mse'] = valid_results[best_method]['test_mse']
    
    return comparison


def create_comparison_plots(all_results: Dict[str, Any], save_folder: str) -> None:
    """
    Создаёт сравнительные графики для всех методов.
    
    Args:
        all_results: словарь с результатами
        save_folder: папка для сохранения
    """

    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid results to plot")
        return
    

    colors = {
        'baseline': '#1f77b4',
        'sobol': '#ff7f0e',
        'gradnorm': '#2ca02c',
        'uncertainty_weighting': '#d62728'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 Model Loss
    ax = axes[0, 0]
    for method, result in valid_results.items():
        history = result['history']
        epochs = range(1, len(history['model_loss']) + 1)
        color = colors.get(method, '#333333')
        ax.plot(epochs, history['model_loss'], label=method, color=color, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Model Loss')
    ax.set_title('Model Loss Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for method, result in valid_results.items():
        history = result['history']
        epochs = range(1, len(history['graph_loss']) + 1)
        color = colors.get(method, '#333333')
        ax.plot(epochs, history['graph_loss'], label=method, color=color, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Graph Loss')
    ax.set_title('Graph Loss Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for method, result in valid_results.items():
        history = result['history']
        if history['val_loss']:
            epochs = range(1, len(history['val_loss']) + 1)
            color = colors.get(method, '#333333')
            ax.plot(epochs, history['val_loss'], label=method, color=color, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for method, result in valid_results.items():
        history = result['history']
        epochs = range(1, len(history['lam_graph']) + 1)
        color = colors.get(method, '#333333')
        ax.plot(epochs, history['lam_graph'], label=method, color=color, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('λ_graph')
    ax.set_title('Lambda Graph Values Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'comparison_training.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    methods = list(valid_results.keys())
    x = np.arange(len(methods))
    bar_colors = [colors.get(m, '#333333') for m in methods]

    ax = axes[0]
    mse_values = [valid_results[m]['test_mse'] for m in methods]
    bars = ax.bar(x, mse_values, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, mse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    

    ax = axes[1]
    r2_values = [valid_results[m]['test_r2'] for m in methods]
    bars = ax.bar(x, r2_values, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Test R²')
    ax.set_title('Test R² Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    

    ax = axes[2]
    time_values = [valid_results[m]['training_time_seconds'] for m in methods]
    bars = ax.bar(x, time_values, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, time_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'comparison_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    

    ax = axes[0]
    for method, result in valid_results.items():
        history = result['history']
        epochs = range(1, len(history['lam_nn']) + 1)
        color = colors.get(method, '#333333')
        ax.plot(epochs, history['lam_nn'], label=method, color=color, alpha=0.8, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('λ_nn')
    ax.set_title('Lambda NN Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    

    ax = axes[1]
    for method, result in valid_results.items():
        history = result['history']
        epochs = range(1, len(history['lam_graph']) + 1)
        color = colors.get(method, '#333333')
        ax.plot(epochs, history['lam_graph'], label=method, color=color, alpha=0.8, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('λ_graph')
    ax.set_title('Lambda Graph Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'comparison_lambdas.png'), dpi=150, bbox_inches='tight')
    plt.close()

    if 'baseline' in valid_results:
        baseline_mse = valid_results['baseline']['test_mse']
        other_methods = [m for m in methods if m != 'baseline']
        
        if other_methods:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            improvements = []
            for m in other_methods:
                imp = (baseline_mse - valid_results[m]['test_mse']) / baseline_mse * 100
                improvements.append(imp)
            
            x = np.arange(len(other_methods))
            bar_colors_other = [colors.get(m, '#333333') for m in other_methods]
            bars = ax.bar(x, improvements, color=bar_colors_other, alpha=0.8)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(other_methods, rotation=45, ha='right')
            ax.set_ylabel('Improvement over Baseline (%)')
            ax.set_title('MSE Improvement over Baseline (lambda=0)')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, improvements):
                va = 'bottom' if val >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.2f}%', ha='center', va=va, fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, 'comparison_improvement.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"\nComparison plots saved to: {save_folder}")


def create_latex_table(all_results: Dict[str, Any]) -> str:

    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Comparison of Lambda Optimization Methods}
\label{tab:lambda_comparison}
\begin{tabular}{lcccc}
\toprule
Method & Test MSE & Test R² & Time (s) & Best Epoch \\
\midrule
"""
    
    for method, result in valid_results.items():
        latex += f"{method} & {result['test_mse']:.6f} & {result['test_r2']:.4f} & "
        latex += f"{result['training_time_seconds']:.1f} & {result['best_epoch']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def print_latex_table(all_results: Dict[str, Any]) -> None:
    print("\n" + "="*60)
    print("LATEX TABLE (for your report):")
    print("="*60)
    print(create_latex_table(all_results))
