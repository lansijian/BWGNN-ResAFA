from dgl.data.utils import load_graphs
import dgl
import dgl.function as fn
import torch
import warnings
import pickle as pkl
import numpy as np
import os
from dgl.nn.pytorch.conv import EdgeWeightNorm
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import scipy.sparse as sp

warnings.filterwarnings("ignore")

def load_mat_dataset(name, data_path):
    """
    从本地.mat文件加载Amazon或Yelp数据集并构建DGL图
    
    【重要修复】不再创建train/val/test mask，因为原始代码在train()函数中使用train_test_split统一划分
    这样保证所有数据集使用相同的划分逻辑和比例
    """
    prefix = data_path if data_path else './data'
    
    if name == 'amazon':
        mat_file = os.path.join(prefix, 'Amazon.mat')
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"Amazon.mat not found at {mat_file}")
        data = loadmat(mat_file)
        
        # 提取数据
        features = data['features']
        labels = data['label'].flatten()
        homo_adj = data.get('homo', None)
        net_upu = data.get('net_upu', None)
        net_usu = data.get('net_usu', None)
        net_uvu = data.get('net_uvu', None)
        
        # 构建异构图（包含多个关系）
        data_dict = {}
        if net_upu is not None and sp.issparse(net_upu) and net_upu.nnz > 0:
            net_upu = net_upu.tocoo()
            data_dict[('review', 'upu', 'review')] = (torch.LongTensor(net_upu.row), torch.LongTensor(net_upu.col))
        if net_usu is not None and sp.issparse(net_usu) and net_usu.nnz > 0:
            net_usu = net_usu.tocoo()
            data_dict[('review', 'usu', 'review')] = (torch.LongTensor(net_usu.row), torch.LongTensor(net_usu.col))
        if net_uvu is not None and sp.issparse(net_uvu) and net_uvu.nnz > 0:
            net_uvu = net_uvu.tocoo()
            data_dict[('review', 'uvu', 'review')] = (torch.LongTensor(net_uvu.row), torch.LongTensor(net_uvu.col))
        
        # 如果没有关系图，使用同构图
        if len(data_dict) == 0 and homo_adj is not None:
            homo_adj = homo_adj.tocoo() if sp.issparse(homo_adj) else csr_matrix(homo_adj).tocoo()
            data_dict[('review', 'homo', 'review')] = (torch.LongTensor(homo_adj.row), torch.LongTensor(homo_adj.col))
        
        if len(data_dict) == 0:
            raise ValueError("No valid adjacency matrix found in Amazon.mat")
        
        graph = dgl.heterograph(data_dict, num_nodes_dict={'review': len(labels)})
        
        # 设置节点特征和标签
        if sp.issparse(features):
            features = torch.FloatTensor(features.todense())
        else:
            features = torch.FloatTensor(features)
        graph.ndata['feature'] = features
        graph.ndata['label'] = torch.LongTensor(labels)
        
    elif name == 'yelp':
        mat_file = os.path.join(prefix, 'YelpChi.mat')
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"YelpChi.mat not found at {mat_file}")
        data = loadmat(mat_file)
        
        # 提取数据
        features = data['features']
        labels = data['label'].flatten()
        homo_adj = data.get('homo', None)
        net_rur = data.get('net_rur', None)
        net_rtr = data.get('net_rtr', None)
        net_rsr = data.get('net_rsr', None)
        
        # 构建异构图（包含多个关系）
        data_dict = {}
        if net_rur is not None and sp.issparse(net_rur) and net_rur.nnz > 0:
            net_rur = net_rur.tocoo()
            data_dict[('review', 'rur', 'review')] = (torch.LongTensor(net_rur.row), torch.LongTensor(net_rur.col))
        if net_rtr is not None and sp.issparse(net_rtr) and net_rtr.nnz > 0:
            net_rtr = net_rtr.tocoo()
            data_dict[('review', 'rtr', 'review')] = (torch.LongTensor(net_rtr.row), torch.LongTensor(net_rtr.col))
        if net_rsr is not None and sp.issparse(net_rsr) and net_rsr.nnz > 0:
            net_rsr = net_rsr.tocoo()
            data_dict[('review', 'rsr', 'review')] = (torch.LongTensor(net_rsr.row), torch.LongTensor(net_rsr.col))
        
        # 如果没有关系图，使用同构图
        if len(data_dict) == 0 and homo_adj is not None:
            homo_adj = homo_adj.tocoo() if sp.issparse(homo_adj) else csr_matrix(homo_adj).tocoo()
            data_dict[('review', 'homo', 'review')] = (torch.LongTensor(homo_adj.row), torch.LongTensor(homo_adj.col))
        
        if len(data_dict) == 0:
            raise ValueError("No valid adjacency matrix found in YelpChi.mat")
        
        graph = dgl.heterograph(data_dict, num_nodes_dict={'review': len(labels)})
        
        # 设置节点特征和标签
        if sp.issparse(features):
            features = torch.FloatTensor(features.todense())
        else:
            features = torch.FloatTensor(features)
        graph.ndata['feature'] = features
        graph.ndata['label'] = torch.LongTensor(labels)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    # 【修复】不再创建train/val/test mask
    # 原始代码在train()函数中使用train_test_split统一划分所有数据集
    # 这样保证所有数据集使用相同的划分逻辑和比例
    
    return graph

def load_prediction_file(name, load_epoch, homo):
    """加载预测文件，优先从models文件夹读取，兼容旧格式"""
    homo_str = "Homo" if homo else "Hetero"
    homo_num = 1 if homo else 0
    
    # 优先尝试从models文件夹读取新的best_probs格式
    best_probs_file = f'models/best_probs_{name}_BWGNN_{homo_str}.pkl'
    if os.path.exists(best_probs_file):
        return best_probs_file
    
    # 向后兼容：尝试从当前目录读取新的best_probs格式
    best_probs_file_old = f'best_probs_{name}_BWGNN_{homo_str}.pkl'
    if os.path.exists(best_probs_file_old):
        return best_probs_file_old
    
    # 尝试旧格式（特定epoch的预测文件）
    old_format_file = f'probs_{name}_BWGNN_{load_epoch}_{homo_num}.pkl'
    if os.path.exists(old_format_file):
        return old_format_file
    
    # 如果都不存在，返回best_probs格式的文件名（用于错误提示）
    return f'models/best_probs_{name}_BWGNN_{homo_str}.pkl'

class Dataset:
    def __init__(self, load_epoch, name='tfinance', del_ratio=0., homo=True, data_path='', adj_type='sym', train_ratio=0.4):
        """
        数据集加载类
        
        Args:
            load_epoch: 加载哪个epoch的预测结果（用于边删除）
            name: 数据集名称
            del_ratio: 删除边的比例
            homo: 是否使用同构图
            data_path: 数据路径
            adj_type: 邻接矩阵类型
            train_ratio: 训练集比例（对于yelp和amazon数据集有效，默认0.4）
        """
        self.name = name
        graph = None
        prefix = data_path
        if name == 'tfinance':
            graph, label_dict = load_graphs(f'{prefix}/tfinance/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)
            if del_ratio != 0.:
                graph = graph.add_self_loop()
                pkl_file = load_prediction_file('tfinance', load_epoch, homo)
                if not os.path.exists(pkl_file):
                    raise FileNotFoundError(f"Prediction file {pkl_file} not found. Please run with del_ratio=0 first to generate best_probs_tfinance_BWGNN_{'Homo' if homo else 'Hetero'}.pkl")
                with open(pkl_file, 'rb') as f:
                    pred_y = pkl.load(f)
                    graph.ndata['pred_y'] = pred_y
                graph = random_walk_update(graph, del_ratio, adj_type)
                graph = dgl.remove_self_loop(graph)
            else:
                graph = graph.add_self_loop()

        elif name == 'tsocial':
            graph, label_dict = load_graphs(f'{prefix}/tsocial/tsocial')
            graph = graph[0]
            if del_ratio != 0.:
                graph = graph.add_self_loop()
                pkl_file = load_prediction_file('tsocial', load_epoch, homo)
                if not os.path.exists(pkl_file):
                    raise FileNotFoundError(f"Prediction file {pkl_file} not found. Please run with del_ratio=0 first to generate best_probs_tsocial_BWGNN_{'Homo' if homo else 'Hetero'}.pkl")
                with open(pkl_file, 'rb') as f:
                    pred_y = pkl.load(f)
                    graph.ndata['pred_y'] = pred_y
                graph = random_walk_update(graph, del_ratio, adj_type)
                graph = dgl.remove_self_loop(graph)
            else:
                graph = graph.add_self_loop()

        elif name == 'yelp':
            # 从本地.mat文件加载
            mat_file = os.path.join(prefix, 'YelpChi.mat')
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f"YelpChi.mat not found at {mat_file}. Please ensure the file exists in data/ directory.")
            print(f"Loading Yelp dataset from local file: {mat_file}")
            # 【修复】不再在dataset.py中创建mask，统一在train()函数中使用train_test_split划分
            graph = load_mat_dataset('yelp', prefix)
            if homo:
                # 【修复】不再传递train_mask等，因为不再创建这些mask
                graph = dgl.to_homogeneous(graph, ndata=['feature', 'label'])
                graph = dgl.add_self_loop(graph)
                if del_ratio != 0.:
                    pkl_file = load_prediction_file('yelp', load_epoch, homo)
                    if not os.path.exists(pkl_file):
                        raise FileNotFoundError(f"Prediction file {pkl_file} not found. Please run with del_ratio=0 first to generate best_probs_yelp_BWGNN_Homo.pkl")
                    with open(pkl_file, 'rb') as f:
                        pred_y = pkl.load(f)
                        graph.ndata['pred_y'] = pred_y
                    graph = random_walk_update(graph, del_ratio, adj_type)
                    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
            else:
                if del_ratio != 0.:
                    pkl_file = load_prediction_file('yelp', load_epoch, homo)
                    if not os.path.exists(pkl_file):
                        raise FileNotFoundError(f"Prediction file {pkl_file} not found. Please run with del_ratio=0 first to generate best_probs_yelp_BWGNN_Hetero.pkl")
                    with open(pkl_file, 'rb') as f:
                        pred_y = pkl.load(f)
                    data_dict = {}
                    flag = 1
                    for relation in graph.canonical_etypes:
                        # 【修复】不再传递train_mask等，因为不再创建这些mask
                        graph_r = dgl.to_homogeneous(graph[relation], ndata=['feature', 'label'])
                        graph_r = dgl.add_self_loop(graph_r)
                        graph_r.ndata['pred_y'] = pred_y
                        graph_r = random_walk_update(graph_r, del_ratio, adj_type)
                        graph_r = dgl.remove_self_loop(graph_r)
                        data_dict[('review', str(flag), 'review')] = graph_r.edges()
                        flag += 1
                    graph_new = dgl.heterograph(data_dict) 
                    graph_new.ndata['label'] = graph.ndata['label']
                    graph_new.ndata['feature'] = graph.ndata['feature']
                    # 【修复】不再设置train_mask等，因为不再创建这些mask
                    graph = graph_new


        
        elif name == 'amazon':
            # 从本地.mat文件加载
            mat_file = os.path.join(prefix, 'Amazon.mat')
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f"Amazon.mat not found at {mat_file}. Please ensure the file exists in data/ directory.")
            print(f"Loading Amazon dataset from local file: {mat_file}")
            # 【修复】不再在dataset.py中创建mask，统一在train()函数中使用train_test_split划分
            # 注意：Amazon数据集在train()函数中会跳过前3305个节点（原始代码的特殊处理）
            graph = load_mat_dataset('amazon', prefix)
            if homo:
                # 【修复】不再传递train_mask等，因为不再创建这些mask
                graph = dgl.to_homogeneous(graph, ndata=['feature', 'label'])
                graph = dgl.add_self_loop(graph)
                if del_ratio != 0.:
                    pkl_file = load_prediction_file('amazon', load_epoch, homo)
                    if not os.path.exists(pkl_file):
                        raise FileNotFoundError(f"Prediction file {pkl_file} not found. Please run with del_ratio=0 first to generate best_probs_amazon_BWGNN_Homo.pkl")
                    with open(pkl_file, 'rb') as f:
                        pred_y = pkl.load(f)
                        graph.ndata['pred_y'] = pred_y
                    graph = random_walk_update(graph, del_ratio, adj_type)
                    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
            else:
                if del_ratio != 0.:
                    pkl_file = load_prediction_file('amazon', load_epoch, homo)
                    if not os.path.exists(pkl_file):
                        raise FileNotFoundError(f"Prediction file {pkl_file} not found. Please run with del_ratio=0 first to generate best_probs_amazon_BWGNN_Hetero.pkl")
                    with open(pkl_file, 'rb') as f:
                        pred_y = pkl.load(f)
                    data_dict = {}
                    flag = 1
                    for relation in graph.canonical_etypes:
                        graph[relation].ndata['pred_y'] = pred_y
                        graph_r = dgl.add_self_loop(graph[relation])
                        graph_r = random_walk_update(graph_r, del_ratio, adj_type)
                        graph_r = dgl.remove_self_loop(graph_r)
                        data_dict[('review', str(flag), 'review')] = graph_r.edges()
                        flag += 1
                    graph_new = dgl.heterograph(data_dict) 
                    graph_new.ndata['label'] = graph.ndata['label']
                    graph_new.ndata['feature'] = graph.ndata['feature']
                    # 【修复】不再设置train_mask等，因为不再创建这些mask
                    graph = graph_new
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph

def random_walk_update(graph, delete_ratio, adj_type):
    edge_weight = torch.ones(graph.num_edges())
    if adj_type == 'sym':
        norm = EdgeWeightNorm(norm='both')
    else:
        norm = EdgeWeightNorm(norm='left')
    graph.edata['w'] = norm(graph, edge_weight)
    # functions
    aggregate_fn = fn.u_mul_e('h', 'w', 'm')
    reduce_fn = fn.sum(msg='m', out='ay')

    graph.ndata['h'] = graph.ndata['pred_y']
    graph.update_all(aggregate_fn, reduce_fn)
    graph.ndata['ly'] = graph.ndata['pred_y'] - graph.ndata['ay']
    graph.apply_edges(inner_product_black)
    black = graph.edata['inner_black']
    threshold = int(delete_ratio * graph.num_edges())
    edge_to_move = set(black.sort()[1][:threshold].tolist())
    edge_to_protect = set()
    graph_new = dgl.remove_edges(graph, list(edge_to_move.difference(edge_to_protect)))
    return graph_new

def inner_product_black(edges):
    return {'inner_black': (edges.src['ly'] * edges.dst['ly']).sum(axis=1)}
