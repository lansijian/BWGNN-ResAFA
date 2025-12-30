"""
独立的后处理可视化脚本
从保存的训练指标JSON文件生成可视化图表

使用方法:
    python visualize_training.py --dataset tfinance --homo 1
    或
    python visualize_training.py --metrics_file training_metrics_tfinance_BWGNN_Homo.json
"""

import argparse
import json
import os
import sys

# 尝试导入matplotlib
try:
    import matplotlib
    # 使用非交互式后端，避免显示问题
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"Error: matplotlib is not available ({e})")
    print("Please install matplotlib to use visualization:")
    print("  pip install matplotlib")
    print("  or")
    print("  pip uninstall Pillow && pip install Pillow")
    sys.exit(1)
except Exception as e:
    error_msg = str(e)
    if 'DLL' in error_msg or '_imaging' in error_msg or 'Pillow' in error_msg:
        print("\n" + "="*60)
        print("ERROR: Pillow DLL loading failed")
        print("="*60)
        print("This is a common Windows issue. Please run:")
        print("  pip uninstall Pillow -y")
        print("  pip install Pillow")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print(f"Error: matplotlib initialization failed ({e})")
        sys.exit(1)


def load_metrics(metrics_file=None, dataset=None, homo=None, model='BWGNN'):
    """加载训练指标数据（从metrics文件夹）"""
    if metrics_file:
        # 如果直接指定文件路径，尝试不同位置
        possible_paths = [
            metrics_file,  # 直接指定的路径
            f'metrics/{metrics_file}',  # metrics文件夹中
            metrics_file if os.path.isabs(metrics_file) else os.path.join('metrics', metrics_file)
        ]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        print(f"Error: Metrics file not found: {metrics_file}")
    elif dataset and homo is not None:
        model = model.upper()
        graph_type = 'Homo' if homo else 'Hetero'
        # 优先从metrics文件夹读取
        metrics_file = f'metrics/training_metrics_{dataset}_{model}_{graph_type}.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        # 向后兼容：尝试旧命名（无模型名前缀）
        fallback_metrics = f'metrics/training_metrics_{dataset}_BWGNN_{graph_type}.json'
        if os.path.exists(fallback_metrics):
            with open(fallback_metrics, 'r', encoding='utf-8') as f:
                return json.load(f)
        legacy_local = f'training_metrics_{dataset}_BWGNN_{graph_type}.json'
        if os.path.exists(legacy_local):
            print(f"Warning: Found old format metrics file: {legacy_local}")
            print("  Consider moving it to metrics/ folder for better organization.")
            with open(legacy_local, 'r', encoding='utf-8') as f:
                return json.load(f)
        legacy_no_prefix = f'training_metrics_{dataset}_{graph_type}.json'
        if os.path.exists(legacy_no_prefix):
            with open(legacy_no_prefix, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 文件不存在，列出可用的文件
        print(f"Error: Metrics file not found: {metrics_file}")
        print("\nAvailable metrics files:")
        # 检查metrics文件夹
        if os.path.exists('metrics'):
            json_files = [f for f in os.listdir('metrics') if f.startswith('training_metrics_') and f.endswith('.json')]
            if json_files:
                print("  In metrics/ folder:")
                for f in json_files:
                    print(f"    - metrics/{f}")
        # 检查当前目录（向后兼容）
        json_files = [f for f in os.listdir('.') if f.startswith('training_metrics_') and f.endswith('.json')]
        if json_files:
            print("  In current directory:")
            for f in json_files:
                print(f"    - {f}")
        if not json_files and not os.path.exists('metrics'):
            print("  (No metrics files found)")
        print("\nPlease run training first to generate metrics file.")
        sys.exit(1)
    else:
        print("Error: Please specify either --metrics_file or --dataset/--homo[/--model]")
        print("Usage:")
        print("  python visualize_training.py --dataset tfinance --homo 1")
        print("  python visualize_training.py --metrics_file training_metrics_tfinance_BWGNN_Homo.json")
        sys.exit(1)


def visualize_metrics(metrics_data):
    """从指标数据生成可视化图表"""
    dataset = metrics_data['dataset']
    model_name = metrics_data.get('model_name', 'BWGNN')
    graph_type = metrics_data.get('graph_type', metrics_data.get('model_type', 'Homo'))
    model_label = f'{model_name}_{graph_type}' if graph_type else model_name
    best_epoch = metrics_data['best_epoch']
    epochs = metrics_data['epochs']
    train_losses = metrics_data['train_losses']
    val_losses = metrics_data['val_losses']
    val_f1s = metrics_data['val_f1s']
    val_aucs = metrics_data.get('val_aucs', [])  # 向后兼容
    learning_rates = metrics_data.get('learning_rates', [])  # 向后兼容
    test_recalls = metrics_data['test_recalls']
    test_precisions = metrics_data['test_precisions']
    test_f1s = metrics_data['test_f1s']
    test_aucs = metrics_data['test_aucs']
    
    # 创建figures目录（如果不存在）
    os.makedirs('figures', exist_ok=True)
    
    # 设置中文字体和参数
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.max_open_warning'] = 0
    
    print(f"\nGenerating visualizations for {dataset} ({model_label})...")
    print(f"Best epoch: {best_epoch}")
    print(f"Total epochs: {len(epochs)}")
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Metrics - Dataset: {dataset} | Model: {model_label} | Best Epoch: {best_epoch}',
                 fontsize=14, fontweight='bold')
    
    # 1. Loss曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. F1和AUC曲线（更新为验证集AUC）
    ax2 = axes[0, 1]
    if val_aucs:
        ax2.plot(epochs, val_aucs, 'b-', label='Val AUC', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, val_f1s, 'g-', label='Val F1', linewidth=2)
        ax2_twin.plot(epochs, test_f1s, 'orange', label='Test F1', linewidth=2, linestyle='--')
        ax2_twin.set_ylabel('F1 Score', fontsize=11, color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2.set_ylabel('AUC', fontsize=11, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_title('Validation AUC and F1 Score', fontsize=12, fontweight='bold')
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
    else:
        # 向后兼容：如果没有val_aucs，使用原来的显示方式
        ax2.plot(epochs, val_f1s, 'g-', label='Val F1', linewidth=2)
        ax2.plot(epochs, test_f1s, 'orange', label='Test F1', linewidth=2)
        ax2.set_ylabel('F1 Score', fontsize=11)
        ax2.set_title('Validation and Test F1 Score', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
    ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Recall和Precision曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, test_recalls, 'purple', label='Test Recall', linewidth=2, marker='o', markersize=3)
    ax3.plot(epochs, test_precisions, 'brown', label='Test Precision', linewidth=2, marker='s', markersize=3)
    ax3.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Test Recall and Precision', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    # 4. AUC曲线
    ax4 = axes[1, 1]
    ax4.plot(epochs, test_aucs, 'c-', label='Test AUC', linewidth=2, marker='^', markersize=3)
    ax4.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('AUC', fontsize=11)
    ax4.set_title('Test AUC Score', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    # 保存图片（文件名包含数据集信息）
    save_path = f'figures/training_metrics_{dataset}_{model_label}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'  -> Training metrics visualization saved to {save_path}')
    plt.close()
    
    # 创建单独的指标汇总图
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(epochs, test_recalls, 'purple', label='Recall', linewidth=2)
    ax.plot(epochs, test_precisions, 'brown', label='Precision', linewidth=2)
    ax.plot(epochs, test_f1s, 'orange', label='F1 Score', linewidth=2)
    ax.plot(epochs, test_aucs, 'c-', label='AUC', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, linewidth=2, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Test Set Performance Metrics - {dataset} ({model_label})',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path2 = f'figures/test_metrics_summary_{dataset}_{model_label}.png'
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f'  -> Test metrics summary saved to {save_path2}')
    plt.close()
    
    # 显示最佳指标
    best_metrics = metrics_data.get('best_metrics', {})
    print(f'\n  -> Visualization completed successfully!')
    print(f'\nBest metrics at epoch {best_epoch}:')
    if best_metrics:
        print(f'  Loss: {best_metrics.get("best_loss", 0):.4f}')
        print(f'  Recall: {best_metrics.get("final_trec", 0)*100:.2f}%')
        print(f'  Precision: {best_metrics.get("final_tpre", 0)*100:.2f}%')
        print(f'  F1 Score: {best_metrics.get("final_tmf1", 0)*100:.2f}%')
        print(f'  AUC: {best_metrics.get("final_tauc", 0)*100:.2f}%')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training metrics from saved JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用数据集名称和模型类型
  python visualize_training.py --dataset tfinance --homo 1 --model BWGNN
  
  # 直接指定JSON文件
  python visualize_training.py --metrics_file training_metrics_tfinance_BWGNN_Homo.json
  
  # 查看所有可用的指标文件
  python visualize_training.py --dataset tfinance --homo 1  # (会显示可用文件列表)
        """
    )
    parser.add_argument('--metrics_file', type=str, default=None,
                        help='Path to metrics JSON file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (tfinance/tsocial/yelp/amazon)')
    parser.add_argument('--homo', type=int, default=None,
                        help='Graph type (1=Homo, 0=Hetero)')
    parser.add_argument('--model', type=str, default='BWGNN',
                        help='Model name (BWGNN/BWGNN_AFA/GCN/GAT/SAGE)')
    
    args = parser.parse_args()
    
    model_name = args.model.upper() if args.model else 'BWGNN'
    # 加载指标数据
    metrics_data = load_metrics(args.metrics_file, args.dataset, args.homo, model=model_name)
    
    # 生成可视化
    visualize_metrics(metrics_data)


if __name__ == '__main__':
    main()

