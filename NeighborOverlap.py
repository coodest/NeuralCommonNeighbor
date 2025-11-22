import argparse
import numpy as np
import torch
import torch.nn.functional as F
# import torch.nn as nn
from torch_sparse import SparseTensor
# import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN
from functools import partial
import sklearn.metrics as skm
# from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from ogb.linkproppred import Evaluator
from torch_geometric.utils import negative_sampling, to_undirected
# from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
from torch_geometric.data import Data
# from ogbdataset import loaddataset
from typing import Iterable

from lp_common import Logger
import pickle
import torch_geometric
import math


context = {"multiclass":False}


def train_test_split_edges(
    data: 'torch_geometric.data.Data',
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
) -> 'torch_geometric.data.Data':
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    .. warning::

        :meth:`~torch_geometric.utils.train_test_split_edges` is deprecated and
        will be removed in a future release.
        Use :class:`torch_geometric.transforms.RandomLinkSplit` instead.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    assert 'batch' not in data  # No batch-mode.

    assert data.num_nodes is not None
    assert data.edge_index is not None

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    del data.edge_index
    del data.edge_attr

    # Return upper triangular portion.
    if not context["multiclass"] or context["predictor"] == "incn1cn1":
        mask = row < col
        row, col = row[mask], col[mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if not context["multiclass"] or context["predictor"] == "incn1cn1":
        if edge_attr is not None:
            out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
            data.train_pos_edge_index, data.train_pos_edge_attr = out
        else:
            data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    else:
        if edge_attr is not None:
            out = data.train_pos_edge_index, edge_attr[n_v + n_t:]
            data.train_pos_edge_index, data.train_pos_edge_attr = out
        else:
            data.train_pos_edge_index = data.train_pos_edge_index
        

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data

# random split dataset
def randomsplit(data: Data, val_ratio: float=0.10, test_ratio: float=0.5):
    def removerepeated(ei):
        if context["predictor"] == "incn1cn1" or context["multiclass"]:
            pass
        else:
            ei = to_undirected(ei)
            ei = ei[:, ei[0]<ei[1]]
        return ei
    data.num_nodes = data.x.shape[0]
    # print(data.edge_index.shape)
    new_data = train_test_split_edges(data, val_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    if not context["multiclass"] or context["predictor"] == "incn1cn1":
        num_val = int(new_data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
        new_data.val_pos_edge_index = new_data.val_pos_edge_index[:, torch.randperm(new_data.val_pos_edge_index.shape[1])]
        split_edge['train']['edge'] = removerepeated(torch.cat((new_data.train_pos_edge_index, new_data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
        split_edge['valid']['edge'] = removerepeated(new_data.val_pos_edge_index[:, -num_val:]).t()
        split_edge['valid']['edge_neg'] = removerepeated(new_data.val_neg_edge_index).t()
        split_edge['test']['edge'] = removerepeated(new_data.test_pos_edge_index).t()
        split_edge['test']['edge_neg'] = removerepeated(new_data.test_neg_edge_index).t()
    else:
        new_data.val_pos_edge_index = new_data.val_pos_edge_index[:, torch.randperm(new_data.val_pos_edge_index.shape[1])]
        split_edge['train']['edge'] = removerepeated(torch.cat((new_data.train_pos_edge_index, new_data.val_pos_edge_index), dim=-1)).t()
        split_edge['valid']['edge'] = removerepeated(new_data.val_pos_edge_index).t()
        split_edge['valid']['edge_neg'] = removerepeated(new_data.val_neg_edge_index).t()
        split_edge['test']['edge'] = removerepeated(new_data.test_pos_edge_index).t()
        split_edge['test']['edge_neg'] = removerepeated(new_data.test_neg_edge_index).t()
    
    return split_edge

def loaddataset(name: str, use_valedges_as_input: bool, load=None):
    with open(f"./input/{name}.pkl", 'rb') as file:
        loaded = pickle.load(file)
        if len(loaded) == 4:
        # if context["multiclass"]:
            x, edge_index, y, edge_label = loaded
            edge_label_dict = dict()
            edge_class_set = set()

            for i in range(len(edge_label)):
                edge_class_set.add(edge_label[i])
            context["num_class"] = len(edge_class_set)

            l2i = dict()
            for index, label in enumerate(edge_class_set):
                l2i[label] = index

            for i in range(len(edge_label)):
                edge_label_dict[(edge_index[0][i], edge_index[1][i])] = l2i[edge_label[i]]
            context["edge_label_dict"] = edge_label_dict
        else:
            x, edge_index, y = loaded


    data = Data(
        x=torch.tensor(x, dtype=torch.float32), 
        edge_index=torch.tensor(edge_index, dtype=torch.int64), 
        y=torch.tensor(y, dtype=torch.int64)
    )
    print(f"raw data edge: {data.edge_index.shape}")
    split_edge = randomsplit(data)
    if not context["multiclass"] or context["predictor"] == "incn1cn1":
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    # data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    data.edge_index = split_edge["train"]["edge"].t()
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    # else:
    #     dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
    #     split_edge = dataset.get_edge_split()
    #     data = dataset[0]
    #     edge_index = data.edge_index
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    # if name == "ppa":
    #     data.x = torch.argmax(data.x, dim=-1)
    #     data.max_x = torch.max(data.x).item()
    # elif name == "ddi":
    #     data.x = torch.arange(data.num_nodes)
    #     data.max_x = data.num_nodes
    # if load is not None:
    #     data.x = torch.load(load, map_location="cpu")
    #     data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])


    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge






def metric(pos_score, neg_score, pos_label):
    results = {}
    
    if context["multiclass"]:
        all_scores = pos_score
        all_labels_np = np.array(pos_label)

        all_scores_np = torch.sigmoid(all_scores)
        all_preds = torch.argmax(all_scores_np, axis=1).cpu().numpy()
    else:
        all_scores = torch.cat([pos_score, neg_score], dim=0)
        all_labels = torch.cat([pos_label, torch.zeros(neg_score.size(0))], dim=0)
        all_labels_np = all_labels.cpu().numpy()

        all_scores_np = torch.sigmoid(all_scores).cpu().numpy()
        all_preds = (all_scores_np > 0.5).astype(int)

    predicted, score, ground_truth = all_preds, all_scores_np, all_labels_np
    if len(predicted) == 0:
        Logger.log("predicted value is empty.")
        return

    if context["multiclass"]:
        # accuracy
        accuracy = skm.accuracy_score(ground_truth, predicted)

        labels = set()
        for e in ground_truth:
            labels.add(e)

        # Micro-F1
        micro_f1 = skm.f1_score(ground_truth, predicted, labels=list(labels), average="micro")

        # Macro-F1
        macro_f1 = skm.f1_score(ground_truth, predicted, labels=list(labels), average="macro")

        # Logger.log("Acc: {:.4f} Micro-F1: {:.4f} Macro-F1: {:.4f}".format(accuracy, micro_f1, macro_f1))
        results['accuracy'], results['micro_f1'], results['macro_f1'] = accuracy, micro_f1, macro_f1
    else:
        # auc
        auc = skm.roc_auc_score(ground_truth, score)

        # accuracy
        accuracy = skm.accuracy_score(ground_truth, predicted)

        # recall
        recall = skm.recall_score(ground_truth, predicted)

        # precision
        precision = skm.precision_score(ground_truth, predicted)

        # F1
        f1 = skm.f1_score(ground_truth, predicted)

        # AUPR
        pr, re, _ = skm.precision_recall_curve(ground_truth, score)
        aupr = skm.auc(re, pr)

        # AP
        ap = skm.average_precision_score(ground_truth, score)

        # Logger.log("Acc: {:.4f} AUC: {:.4f} Pr: {:.4f} Re: {:.4f} F1: {:.4f} AUPR: {:.4f} AP: {:.4f}".format(accuracy, auc, precision, recall, f1, aupr, ap))
        results['accuracy'], results['auc'], results['precision'], results['recall'], results['f1'], results['aupr'], results['ap'] = accuracy, auc, precision, recall, f1, aupr, ap
    return results




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(model,
          predictor,
          data,
          split_edge,
          edge_labels,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    bs = adjmask.shape[0] if batch_size > adjmask.shape[0] else batch_size

    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], bs
    ):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)

        if context["multiclass"]:
            pos_label = edge_labels[perm]
            loss = F.cross_entropy(pos_outs, pos_label)
        else:
            pos_losss = -F.logsigmoid(pos_outs).mean()
            edge = negedge[:, perm]
            neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
            neg_losss = -F.logsigmoid(-neg_outs).mean()
            loss = neg_losss + pos_losss

        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    
    preds = []
    pos_train_label = []
    for perm in PermIterator(pos_train_edge.device, pos_train_edge.shape[0], batch_size, False):
        preds.append(predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu())
        if context["multiclass"]:
            for edge in pos_train_edge[perm]:
                if tuple(edge.detach().cpu().numpy()) in context["edge_label_dict"]:
                    pos_train_label.append(context["edge_label_dict"][tuple(edge.detach().cpu().numpy())])
                else:
                    pos_train_label.append(0)
    pos_train_pred = torch.cat(preds, dim=0)

    # pos_train_pred = torch.cat([
    #     predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
    #     for perm in PermIterator(pos_train_edge.device,
    #                              pos_train_edge.shape[0], batch_size, False)
    # ],
    #                            dim=0)

    preds = []
    pos_valid_label = []
    for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False):
        preds.append(predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu())
        if context["multiclass"]:
            for edge in pos_valid_edge[perm]:
                if tuple(edge.detach().cpu().numpy()) in context["edge_label_dict"]:
                    pos_valid_label.append(context["edge_label_dict"][tuple(edge.detach().cpu().numpy())])
                else:
                    pos_valid_label.append(0)
    pos_valid_pred = torch.cat(preds, dim=0)

    # pos_valid_pred = torch.cat([
    #     predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
    #     for perm in PermIterator(pos_valid_edge.device,
    #                              pos_valid_edge.shape[0], batch_size, False)
    # ],
    #                            dim=0)
    

    preds = []
    for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False):
        preds.append(predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu())
    neg_valid_pred = torch.cat(preds, dim=0)


    # neg_valid_pred = torch.cat([
    #     predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
    #     for perm in PermIterator(neg_valid_edge.device,
    #                              neg_valid_edge.shape[0], batch_size, False)
    # ],
    #                            dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)


    preds = []
    pos_test_label = []
    for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False):
        preds.append(predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu())
        if context["multiclass"]:
            for edge in pos_test_edge[perm]:
                if tuple(edge.detach().cpu().numpy()) in context["edge_label_dict"]:
                    pos_test_label.append(context["edge_label_dict"][tuple(edge.detach().cpu().numpy())])
                else:
                    pos_test_label.append(0)
    pos_test_pred = torch.cat(preds, dim=0)

    # pos_test_pred = torch.cat([
    #     predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
    #     for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
    #                              batch_size, False)
    # ],
    #                           dim=0)
    
    preds = []
    for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False):
        preds.append(predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu())
    neg_test_pred = torch.cat(preds, dim=0)

    # neg_test_pred = torch.cat([
    #     predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
    #     for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
    #                              batch_size, False)
    # ],
    #                           dim=0)

    results = dict()
    # for K in [20, 50, 100]:
    #     evaluator.K = K

    #     train_hits = evaluator.eval({
    #         'y_pred_pos': pos_train_pred,
    #         'y_pred_neg': neg_valid_pred,
    #     })[f'hits@{K}']

    #     valid_hits = evaluator.eval({
    #         'y_pred_pos': pos_valid_pred,
    #         'y_pred_neg': neg_valid_pred,
    #     })[f'hits@{K}']
    #     test_hits = evaluator.eval({
    #         'y_pred_pos': pos_test_pred,
    #         'y_pred_neg': neg_test_pred,
    #     })[f'hits@{K}']

    #     results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    if context["multiclass"]:
        train_metric = metric(pos_train_pred, neg_valid_pred, pos_train_label)
        valid_metric = metric(pos_valid_pred, neg_valid_pred, pos_valid_label)
        test_metric = metric(pos_test_pred,  neg_test_pred,  pos_test_label)
    else:
        train_metric = metric(pos_train_pred, neg_valid_pred, torch.ones(pos_train_pred.size(0)))
        valid_metric = metric(pos_valid_pred, neg_valid_pred, torch.ones(pos_valid_pred.size(0)))
        test_metric = metric(pos_test_pred,  neg_test_pred,  torch.ones(pos_test_pred.size(0)))

    for key in train_metric:
        results[key] = (train_metric[key], valid_metric[key], test_metric[key])
    
    return results, h.cpu()


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--multiclass', action="store_true", help="whether to use multiclass")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used. 
    parser.add_argument('--cndeg', type=int, default=-1)
    
    # predictor used, such as NCN, NCNC
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")

    
    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    args = parser.parse_args()

    Logger.path = f"output/{args.dataset}-{args.predictor}.log"
    if args.multiclass:
        context["multiclass"] = True

    if args.dataset == "blogcatalog":
        args.runs = 2

    context["predictor"] = args.predictor
    
    return args


def main():
    args = parseargs()
    print(args, flush=True)

    # hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        # ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    # writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    # writer.add_text("hyperparams", hpstr)

    # if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
    #     evaluator = Evaluator(name=f'ogbl-ppa')
    # else:
    #     evaluator = Evaluator(name=f'ogbl-{args.dataset}')
    evaluator = Evaluator(name=f'ogbl-ppa')

    if args.dataset == "blogcatalog":
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    # data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)
    # data = data.to(device)

    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    
    all_results = dict()

    for run in range(0, args.runs):
        print(f"Run: {run + 1}")
        set_seed(run)

        # if args.dataset in ["Cora", "Citeseer", "Pubmed"]:

        data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load) # get a new split of dataset
        data = data.to(device)

        if context["multiclass"]:
            add_zero_class = False
            train_edge_labels = []
            for edge in split_edge['train']['edge']:
                if tuple(edge.numpy()) in context["edge_label_dict"]:
                    train_edge_labels.append(context["edge_label_dict"][tuple(edge.numpy())])
                else:
                    train_edge_labels.append(0)
                    add_zero_class = True
            train_edge_labels = torch.tensor(train_edge_labels, dtype=torch.long)
            if add_zero_class:
                context["num_class"] += 1
        else:
            train_edge_labels = torch.ones_like(split_edge['train']['edge'])
        train_edge_labels = train_edge_labels.to(device)

        bestscore = None
        
        # build model
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
        if context["multiclass"]:
            predictor = predfn(args.hiddim, args.hiddim, context["num_class"], args.nnlayers,
                            args.predp, args.preedp, args.lnnn).to(device)
        else:
            predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                            args.predp, args.preedp, args.lnnn).to(device)
        if args.loadmod:
            keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
        

        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])
        
        for epoch in range(1, 1 + args.epochs):
            if epoch + 1 % 50 == 0:
                print(f"epoch: {epoch}")
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            # t1 = time.time()
            loss = train(model, predictor, data, split_edge, train_edge_labels, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            # print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                # t1 = time.time()
                results, h = test(model, predictor, data, split_edge, evaluator, args.testbs, args.use_valedges_as_input)
                # print(f"test time {time.time()-t1:.2f} s")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}
                # for key, result in results.items():
                #     writer.add_scalars(f"{key}_{run}", {
                #         "trn": result[0],
                #         "val": result[1],
                #         "tst": result[2]
                #     }, epoch)

                # if True:
                for key, result in results.items():
                    train_res, valid_res, test_res = result
                    if valid_res > bestscore[key][1]:
                        bestscore[key] = list(result)
                        if args.save_gemb:
                            torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                        if args.savex:
                            torch.save(model.xemb[0].weight.detach(), f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                        if args.savemod:
                            torch.save(model.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                    # print(key)
                    # print(f'Run: {run + 1:02d}, '
                    #       f'Epoch: {epoch:02d}, '
                    #       f'Loss: {loss:.4f}, '
                    #       f'Train: {100 * train_hits:.2f}%, '
                    #       f'Valid: {100 * valid_hits:.2f}%, '
                    #       f'Test: {100 * test_hits:.2f}%')
        print('---', flush=True)
        # print(f"best {bestscore}")
        # if args.dataset == "collab":
        #     ret.append(bestscore["Hits@50"][-2:])
        # elif args.dataset == "ppa":
        #     ret.append(bestscore["Hits@100"][-2:])
        # elif args.dataset == "ddi":
        #     ret.append(bestscore["Hits@20"][-2:])
        # elif args.dataset == "citation2":
        #     ret.append(bestscore[-2:])
        # elif args.dataset in ["Pubmed", "Cora", "Citeseer"]:
        #     ret.append(bestscore["Hits@100"][-2:])
        # else:
            # raise NotImplementedError
        for key in bestscore:
            if key not in all_results:
                all_results[key] = list()
            all_results[key].append(bestscore[key][-2:])  # store only val and test, no train

    # print(ret)
    for key in all_results:
        ret = np.array(all_results[key])
        # out = f"{key}: val {np.average(ret[:, 0]):.4f} +- {np.std(ret[:, 0]):.4f} ||| tst {np.average(ret[:, 1]):.4f} +- {np.std(ret[:, 1]):.4f}"
        score = np.average(ret[:, 1])
        std_dev = np.std(ret[:, 1])
        Logger.log(f"{key}: tst {score:.4f} +- {std_dev:.4f}")
    Logger.log("--- fin of test ---\n\n\n")


if __name__ == "__main__":
    main()