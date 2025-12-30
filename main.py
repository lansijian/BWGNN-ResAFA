import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import pickle as pkl
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split

from BWGNN import *
from baselines import GCN, GAT, GraphSAGE
from dataset import Dataset

# 导入可视化库
SUPPORTED_MODELS = ['BWGNN', 'BWGNN_AFA', 'BWGNN_HETERO_AFA', 
                    'BWGNN_AFA_NoResidual', 'BWGNN_AFA_NoLayerNorm', 'BWGNN_AFA_Sum',
                    'BWGNN_HETERO_AFA_NoResidual', 'BWGNN_HETERO_AFA_NoLayerNorm', 'BWGNN_HETERO_AFA_Sum',
                    'GCN', 'GAT', 'SAGE']

MATPLOTLIB_AVAILABLE = False
plt = None
try:
    import matplotlib
    # 必须在导入pyplot之前设置后端
    matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
    import matplotlib.pyplot as plt
    # 验证导入是否成功
    if plt is not None:
        MATPLOTLIB_AVAILABLE = True
    else:
        print("Warning: matplotlib.pyplot is None")
except ImportError as e:
    print(f"Warning: matplotlib import failed ({e})")
    print("Please install matplotlib: pip install matplotlib")
except Exception as e:
    error_msg = str(e)
    # 检查是否是Pillow的DLL加载问题
    if 'DLL' in error_msg or '_imaging' in error_msg or 'Pillow' in error_msg:
        # Pillow DLL问题，静默处理，稍后在函数中处理
        pass
    else:
        print(f"Warning: matplotlib initialization failed ({e})")
        import traceback
        traceback.print_exc()


def forward_with_graph(model, graph, features, model_name):
    """
    Helper to standardize forward calls across models that either store
    the graph internally (BWGNN family) or require it explicitly (GCN/GAT/SAGE).
    """
    if model_name in {'GCN', 'GAT', 'SAGE'}:
        return model(graph, features)
    return model(features)


def build_model(args, graph, in_feats, h_feats, num_classes):
    """Factory for all supported models."""
    model_name = args.model
    if model_name == 'BWGNN':
        if args.homo:
            return BWGNN(in_feats, h_feats, num_classes, graph, d=args.order)
        return BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    if model_name == 'BWGNN_AFA':
        if not args.homo:
            raise ValueError("BWGNN_AFA 当前仅支持同构图（--homo 1）。")
        return BWGNN_AFA(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    if model_name == 'BWGNN_HETERO_AFA':
        if args.homo:
            raise ValueError("BWGNN_HETERO_AFA 仅适用于异构图，请设置 --homo 0。")
        return BWGNN_Hetero_AFA(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    # 消融实验变体：同构版
    if model_name == 'BWGNN_AFA_NoResidual':
        if not args.homo:
            raise ValueError("BWGNN_AFA_NoResidual 仅支持同构图（--homo 1）。")
        return BWGNN_AFA_NoResidual(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    if model_name == 'BWGNN_AFA_NoLayerNorm':
        if not args.homo:
            raise ValueError("BWGNN_AFA_NoLayerNorm 仅支持同构图（--homo 1）。")
        return BWGNN_AFA_NoLayerNorm(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    if model_name == 'BWGNN_AFA_Sum':
        if not args.homo:
            raise ValueError("BWGNN_AFA_Sum 仅支持同构图（--homo 1）。")
        return BWGNN_AFA_Sum(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    # 消融实验变体：异构版
    if model_name == 'BWGNN_HETERO_AFA_NoResidual':
        if args.homo:
            raise ValueError("BWGNN_HETERO_AFA_NoResidual 仅适用于异构图，请设置 --homo 0。")
        return BWGNN_Hetero_AFA_NoResidual(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    if model_name == 'BWGNN_HETERO_AFA_NoLayerNorm':
        if args.homo:
            raise ValueError("BWGNN_HETERO_AFA_NoLayerNorm 仅适用于异构图，请设置 --homo 0。")
        return BWGNN_Hetero_AFA_NoLayerNorm(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    if model_name == 'BWGNN_HETERO_AFA_Sum':
        if args.homo:
            raise ValueError("BWGNN_HETERO_AFA_Sum 仅适用于异构图，请设置 --homo 0。")
        return BWGNN_Hetero_AFA_Sum(in_feats, h_feats, num_classes, graph, d=args.order, dropout=args.dropout)

    if model_name in {'GCN', 'GAT', 'SAGE'} and not args.homo:
        raise ValueError(f"{model_name} 目前仅支持同构图训练，请设置 --homo 1。")

    if model_name == 'GCN':
        return GCN(in_feats, h_feats, num_classes, dropout=args.dropout)

    if model_name == 'GAT':
        return GAT(
            in_feats,
            h_feats,
            num_classes,
            dropout=args.dropout,
            feat_drop=args.dropout,
            attn_drop=args.dropout,
        )

    if model_name == 'SAGE':
        return GraphSAGE(
            in_feats,
            h_feats,
            num_classes,
            dropout=args.dropout,
            feat_drop=args.dropout,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def train(model, g, args, dataset_name):
    """
    训练函数 - 修复：所有数据集都使用train_test_split进行划分（与原始代码一致）
    """
    features = g.ndata['feature']
    labels = g.ndata['label']
    
    # 【核心修复】原始代码对所有数据集都在train()函数中使用train_test_split
    # 而不是使用dataset.py中预创建的mask
    
    # 获取所有节点索引
    index = list(range(len(labels)))
    
    # Amazon数据集特殊处理：跳过前3305个节点（原始代码的特殊处理）
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))
        labels_for_split = labels[index].cpu().numpy()
    else:
        labels_for_split = labels.cpu().numpy()
    
    # 第一步：划分训练集和剩余集
    # 使用train_ratio作为训练集比例
    idx_train, idx_rest, y_train, y_rest = train_test_split(
        index, 
        labels_for_split,
        stratify=labels_for_split,
        train_size=args.train_ratio,
        random_state=2, 
        shuffle=True
    )
    
    # 第二步：将剩余集划分为验证集和测试集
    # 原始代码：test_size=0.67，即验证集占剩余的33%，测试集占剩余的67%
    idx_valid, idx_test, y_valid, y_test = train_test_split(
        idx_rest, 
        y_rest, 
        stratify=y_rest,
        test_size=0.67,  # 原始代码的设置：测试集占剩余的67%，验证集占剩余的33%
        random_state=2, 
        shuffle=True
    )
    
    # 创建mask
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()
    
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    print(f'Data split ratio: train={args.train_ratio:.1%}, val≈{(1-args.train_ratio)*0.33:.1%}, test≈{(1-args.train_ratio)*0.67:.1%}')
    
    # 修复问题2：添加L2正则化（Weight Decay）防止过拟合
    # 对于异常检测这种不平衡数据集，weight_decay=5e-4是常用设置
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    
    # 修复问题3：学习率调度器（可选，论文使用固定学习率）
    # 如果use_scheduler=False，则不使用调度器，保持固定学习率（与论文一致）
    use_scheduler = getattr(args, 'use_scheduler', False)  # 默认False，与论文一致
    if use_scheduler:
        scheduler_step = getattr(args, 'scheduler_step', 30)
        scheduler_gamma = getattr(args, 'scheduler_gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        print(f'Optimizer: Adam(lr={args.lr}, weight_decay={weight_decay})')
        print(f'Scheduler: StepLR(step_size={scheduler_step}, gamma={scheduler_gamma})')
    else:
        # 不使用调度器，固定学习率（论文原实现）
        scheduler = None
        print(f'Optimizer: Adam(lr={args.lr}, weight_decay={weight_decay})')
        print(f'Scheduler: None (fixed learning rate, match paper)')
    
    # 修复问题1：使用验证集AUC或F1作为早停指标，而不是Loss
    # 对于不平衡异常检测，AUC是更可靠的指标
    best_val_auc = 0.0  # 最佳验证集AUC
    best_val_f1 = 0.0   # 最佳验证集F1
    best_f1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0.
    best_loss = float('inf')
    best_epoch = 0
    best_metric_type = 'AUC'  # 记录使用哪个指标保存模型
    
    # 确定使用哪个指标保存模型（在循环外初始化一次即可）
    if args.use_macro_f1_for_save == -1:
        use_macro_f1_for_save = (args.homo == 0)  # 异构图默认用F1，同构图默认用AUC
    else:
        use_macro_f1_for_save = (args.use_macro_f1_for_save == 1)

    # 记录训练指标用于可视化
    train_losses, val_losses, val_f1s, val_aucs = [], [], [], []
    test_recalls, test_precisions, test_f1s, test_aucs = [], [], [], []
    epochs = []
    learning_rates = []  # 记录学习率变化

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    model_name = args.model
    for e in range(1, args.epoch+1):
        model.train()
        logits = forward_with_graph(model, g, features, model_name)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率（如果使用调度器）
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        model.eval()
        val_logits = forward_with_graph(model, g, features, model_name)
        val_loss = F.cross_entropy(val_logits[val_mask], labels[val_mask], weight=torch.tensor([1., weight]))
        probs = val_logits.softmax(1)
        
        # 计算验证集指标
        val_f1, val_thres = get_best_f1(labels[val_mask].cpu().numpy(), probs[val_mask].detach().cpu().numpy())
        val_auc = roc_auc_score(labels[val_mask].cpu().numpy(), probs[val_mask][:, 1].detach().cpu().numpy())
        
        # 计算测试集指标（用于评估，但不用于保存模型）
        preds = torch.zeros_like(labels)
        preds[probs[:, 1] > val_thres] = 1  # 使用验证集找到的最佳阈值
        trec = recall_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy())
        tpre = precision_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy())
        tmf1 = f1_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), average='macro')
        tauc = roc_auc_score(labels[test_mask].cpu().numpy(), probs[test_mask][:, 1].detach().cpu().numpy())
        
        # 记录指标
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_recalls.append(trec)
        test_precisions.append(tpre)
        test_f1s.append(tmf1)
        test_aucs.append(tauc)
        epochs.append(e)

        # 修复：根据论文，应该基于验证集Macro-F1保存模型，而不是AUC
        # 论文："save the model with the best Macro-F1 in validation"
        # use_macro_f1_for_save 已在循环外初始化
        
        model_saved = False
        should_save = False
        
        if use_macro_f1_for_save:
            # 基于验证集Macro-F1保存（论文原实现）
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_auc = val_auc  # 同时更新AUC用于记录
                best_metric_type = 'Macro-F1'
                should_save = True
            elif val_f1 == best_val_f1 and val_auc > best_val_auc:
                # F1相等但AUC提升，也保存
                best_val_auc = val_auc
                best_metric_type = 'Macro-F1+AUC'
                should_save = True
        else:
            # 基于验证集AUC保存（用于同构图或其他场景）
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_f1 = val_f1
                best_metric_type = 'AUC'
                should_save = True
            elif val_auc == best_val_auc and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_metric_type = 'AUC+F1'
                should_save = True
        
        # 【修复】将保存逻辑移到if-else外面，确保两种情况下都能保存
        if should_save:
            best_loss = val_loss.item()
            best_epoch = e
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            pred_y = probs
            model_saved = True
            
            # 保存最佳模型权重和预测结果（统一保存到models文件夹）
            if args.del_ratio == 0:
                graph_type = "Homo" if args.homo else "Hetero"
                model_suffix = f"{args.model}_{graph_type}"
                # 创建models文件夹（如果不存在）
                os.makedirs('models', exist_ok=True)
                
                model_save_path = f'models/best_model_{dataset_name}_{model_suffix}.pth'
                torch.save({
                    'epoch': e,
                    'dataset': dataset_name,
                    'model_type': model_suffix,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if (scheduler is not None and hasattr(scheduler, 'state_dict')) else None,
                    'best_val_auc': best_val_auc,
                    'best_val_f1': best_val_f1,
                    'best_metric_type': best_metric_type,
                    'best_loss': best_loss,
                    'final_trec': final_trec,
                    'final_tpre': final_tpre,
                    'final_tmf1': final_tmf1,
                    'final_tauc': final_tauc,
                    'args': vars(args),
                }, model_save_path)
                
                probs_save_path = f'models/best_probs_{dataset_name}_{model_suffix}.pkl'
                with open(probs_save_path, 'wb') as f:
                    pkl.dump(pred_y, f)
                print(f'  -> Best model saved at epoch {e} (val_auc={val_auc:.4f}, val_f1={val_f1:.4f}) -> {model_save_path}')
        
        # 记录最佳验证F1（仅用于显示）
        if val_f1 > best_f1:
            best_f1 = val_f1
        
        # 输出训练信息，包含学习率变化
        print('Epoch {}/{}: loss={:.4f}, val_loss={:.4f}, val_f1={:.4f}, val_auc={:.4f}, lr={:.6f}, (best_val_f1={:.4f}, best_val_auc={:.4f})'.format(
            e, args.epoch, loss.item(), val_loss.item(), val_f1, val_auc, current_lr, best_f1, best_val_auc))

    time_end = time.time()
    print(f'\nTime cost: {time_end - time_start:.2f}s')
    print(f'Best model at epoch {best_epoch} (metric: {best_metric_type}, val_auc: {best_val_auc:.4f}, val_f1: {best_val_f1:.4f}, val_loss: {best_loss:.4f})')
    
    result = 'REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(
        final_trec*100, final_tpre*100, final_tmf1*100, final_tauc*100)
    with open('result.txt', 'a+') as f:
        f.write(f'[{dataset_name}][{args.model}] {result} (Best epoch: {best_epoch})\n')
    
    # 保存训练指标数据到JSON文件（统一保存到metrics文件夹，用于后处理可视化）
    if args.del_ratio == 0:
        metrics_data = {
            'dataset': dataset_name,
            'model_name': args.model,
            'graph_type': 'Homo' if args.homo else 'Hetero',
            'best_epoch': int(best_epoch),
            'best_metric_type': best_metric_type,
            'epochs': epochs,
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_f1s': [float(x) for x in val_f1s],
            'val_aucs': [float(x) for x in val_aucs],
            'learning_rates': [float(x) for x in learning_rates],
            'test_recalls': [float(x) for x in test_recalls],
            'test_precisions': [float(x) for x in test_precisions],
            'test_f1s': [float(x) for x in test_f1s],
            'test_aucs': [float(x) for x in test_aucs],
            'best_metrics': {
                'best_val_auc': float(best_val_auc),
                'best_val_f1': float(best_val_f1),
                'best_loss': float(best_loss),
                'final_trec': float(final_trec),
                'final_tpre': float(final_tpre),
                'final_tmf1': float(final_tmf1),
                'final_tauc': float(final_tauc),
            }
        }
        # 创建metrics文件夹（如果不存在）
        os.makedirs('metrics', exist_ok=True)
        graph_type = 'Homo' if args.homo else 'Hetero'
        model_suffix = f"{args.model}_{graph_type}"
        metrics_file = f'metrics/training_metrics_{dataset_name}_{model_suffix}.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        print(f'\n  -> Training metrics saved to {metrics_file}')
        print(f'  -> Run visualization: python visualize_training.py --dataset {dataset_name} --homo {args.homo} --model {args.model}')
    
    return final_tmf1, final_tauc


def get_best_f1(labels, probs):
    """寻找最佳F1分数对应的阈值"""
    best_f1, best_thre = 0, 0
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    probs = np.array(probs) if not isinstance(probs, np.ndarray) else probs
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def visualize_metrics(epochs, train_losses, val_losses, val_f1s,
                     test_recalls, test_precisions, test_f1s, test_aucs,
                     dataset_name, homo, best_epoch, model_name='BWGNN'):
    """可视化训练评估指标"""
    # 重新尝试导入matplotlib（以防模块级别的导入失败）
    global MATPLOTLIB_AVAILABLE, plt
    
    if not MATPLOTLIB_AVAILABLE or plt is None:
        # 尝试重新导入
        try:
            # 尝试修复Pillow问题：先尝试重新安装matplotlib的后端
            import sys
            import os
            
            # 如果Pillow有问题，尝试使用不依赖Pillow的后端
            import matplotlib
            # 尝试不同的后端
            backends_to_try = ['Agg', 'SVG', 'PDF']
            
            for backend in backends_to_try:
                try:
                    matplotlib.use(backend, force=True)
                    import matplotlib.pyplot as plt_local
                    plt = plt_local
                    MATPLOTLIB_AVAILABLE = True
                    print(f"  -> Matplotlib re-initialized successfully with {backend} backend")
                    break
                except Exception as backend_error:
                    continue
            
            # 如果所有后端都失败，尝试修复Pillow
            if not MATPLOTLIB_AVAILABLE:
                # 最后一次尝试：强制使用Agg并忽略Pillow错误
                try:
                    import warnings
                    warnings.filterwarnings('ignore', category=UserWarning)
                    # 尝试不使用Pillow的导入方式
                    os.environ['MPLBACKEND'] = 'Agg'
                    import matplotlib
                    matplotlib.use('Agg', force=True)
                    # 尝试延迟导入PIL
                    import matplotlib.pyplot as plt_local
                    plt = plt_local
                    MATPLOTLIB_AVAILABLE = True
                    print("  -> Matplotlib initialized (Pillow issues ignored)")
                except Exception:
                    pass
                    
        except Exception as e:
            error_msg = str(e)
            if 'DLL' in error_msg or '_imaging' in error_msg:
                print("\n" + "="*60)
                print("  ERROR: Pillow DLL loading failed")
                print("="*60)
                print("  This is a common Windows issue with Pillow library.")
                print("  Solution: Please run these commands in your terminal:")
                print()
                print("      pip uninstall Pillow -y")
                print("      pip install Pillow")
                print()
                print("  Or reinstall both matplotlib and Pillow:")
                print()
                print("      pip uninstall matplotlib Pillow -y")
                print("      pip install matplotlib Pillow")
                print()
                print("  After fixing, re-run the training to generate visualizations.")
                print("="*60 + "\n")
            else:
                print(f"  -> Visualization skipped (matplotlib not available): {e}")
                print("  -> Install matplotlib with: pip install matplotlib")
            return
    
    if plt is None:
        print("  -> Visualization skipped (plt is None)")
        return
    
    try:
        graph_type = 'Homo' if homo else 'Hetero'
        model_label = f'{model_name}_{graph_type}'
        os.makedirs('figures', exist_ok=True)
        
        # 设置中文字体和参数
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.max_open_warning'] = 0  # 禁用警告
    
        # 创建2x2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Metrics - {dataset_name} | {model_label} | Best Epoch: {best_epoch}',
                     fontsize=14, fontweight='bold')
        
        # Loss曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # F1分数曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, val_f1s, 'g-', label='Val F1', linewidth=2)
        ax2.plot(epochs, test_f1s, 'orange', label='Test F1', linewidth=2)
        ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('F1 Score', fontsize=11)
        ax2.set_title('F1 Score', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Recall和Precision曲线
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
        
        # AUC曲线
        ax4 = axes[1, 1]
        ax4.plot(epochs, test_aucs, 'c-', label='Test AUC', linewidth=2, marker='^', markersize=3)
        ax4.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('AUC', fontsize=11)
        ax4.set_title('AUC Score', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        save_path = f'figures/training_metrics_{dataset_name}_{model_label}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'  -> Metrics visualization saved to {save_path}')
        plt.close()
        
        # 指标汇总图
        fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(epochs, test_recalls, 'purple', label='Recall', linewidth=2)
        ax.plot(epochs, test_precisions, 'brown', label='Precision', linewidth=2)
        ax.plot(epochs, test_f1s, 'orange', label='F1 Score', linewidth=2)
        ax.plot(epochs, test_aucs, 'c-', label='AUC', linewidth=2)
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Test Set Performance Metrics - {dataset_name} ({model_label})',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path2 = f'figures/test_metrics_summary_{dataset_name}_{model_label}.png'
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f'  -> Metrics summary saved to {save_path2}')
        plt.close()
        
        print(f'  -> Visualization completed successfully for {dataset_name} ({model_label})')
        
    except Exception as e:
        print(f"  -> Error generating visualization: {e}")
        print("  -> Please check matplotlib installation or report the error")
        import traceback
        traceback.print_exc()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, 
                        help="Training ratio (default: 0.4, YelpChi hetero use 0.01 to match paper)")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--del_ratio", type=float, default=0., help="delete ratios")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument("--load_epoch", type=int, default=100, help="load epoch prediction")
    parser.add_argument("--data_path", type=str, default='./data', help="data path")
    parser.add_argument("--model", type=str, default="BWGNN",
                        help="Model to train: BWGNN, BWGNN_AFA, GCN, GAT, SAGE")
    
    # 优化器和调度器参数
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 regularization weight decay (default: 5e-4)")
    parser.add_argument("--scheduler_step", type=int, default=30, help="StepLR step size (default: 30)")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="StepLR gamma (default: 0.5)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (default: 0.0, match paper. Set 0.5 for regularization)")
    parser.add_argument("--use_macro_f1_for_save", type=int, default=-1, help="Use Macro-F1 for model saving (default: -1=auto, 0=use AUC, 1=use F1)")
    parser.add_argument("--use_scheduler", action='store_true', help="Use learning rate scheduler (default: False, match paper. Paper uses fixed lr=0.01)")

    args = parser.parse_args()

    # 兼容大小写：将用户输入的模型名映射到受支持列表
    _model_map = {name.upper(): name for name in SUPPORTED_MODELS}
    model_key = args.model.upper()
    if model_key in _model_map:
        args.model = _model_map[model_key]
    else:
        raise ValueError(f"Unsupported model '{args.model}'. Choose from {SUPPORTED_MODELS}.")
    print(args)
    dataset_name = args.dataset
    del_ratio = args.del_ratio
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    adj_type = args.adj_type
    load_epoch = args.load_epoch
    data_path = args.data_path
    graph = Dataset(load_epoch, dataset_name, del_ratio, homo, data_path, adj_type=adj_type, train_ratio=args.train_ratio).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2
    
    set_random_seed(717)

    def init_model():
        return build_model(args, graph, in_feats, h_feats, num_classes)

    if args.run == 1:
        model = init_model()
        train(model, graph, args, dataset_name)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            model = init_model()
            mf1, auc = train(model, graph, args, dataset_name)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        result = 'MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs))
        print(result)

