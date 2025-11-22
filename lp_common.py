# import numpy as np
# import sklearn.metrics as skm
# from sklearn.linear_model import LogisticRegression
# import random
# import math
import pickle
# import scipy.sparse as sp
# import networkx as nx
# from node2vec import Node2Vec
# from wikipedia2vec import Wikipedia2Vec
# import json
import os
# from torch_geometric.data import Data
# import torch
# import numpy as np
# import random
import os
import pickle
import logging
import datetime
import time
import pytz
import shutil
import psutil
from pathlib import Path

# class Funcs:
#     @staticmethod
#     def rand_int(lower, upper):
#         return random.randint(lower, upper)

#     @staticmethod
#     def shuffle_list(target):
#         return random.shuffle(target)

#     @staticmethod
#     def rand_prob():
#         return random.random()


# class BiMap:
#     def __init__(self):
#         self.index = 0
#         self.forward = dict()
#         self.backward = dict()

#     def inverse(self, index):
#         return self.backward[int(index)]

#     def size(self):
#         return len(self.forward)

#     def get(self, item):
#         if item not in self.forward:
#             self.forward[item] = self.index
#             self.backward[self.index] = item
#             self.index += 1

#         return self.forward[item]



# class GraphData:
#     NO_EDGE = "NO_EDGE"
#     HAS_EDGE = "HAS_EDGE"
#     VIRTUAL_EDGE = "VIRTUAL_EDGE"

#     def __init__(self):
#         # nodes: index of entities
#         self.graph = dict()
#         self.node_map = BiMap()

#         # relations of edges in a graph. useful for knowledge graph
#         self.relations = BiMap()

#         # store the global info of a graph, such as ["features"], ["events"] and ["trajectories"]
#         self.states = dict()


# class Context:
#     """
#     all configurable fields of the project
#     """
#     # 1. loader
#     multi_class = False  # True / False
#     twitter_feat_dim = 500  # the real dimension may be a little bit more than this value
#     embedding_method = 0  # 0: original embeddings / features, 1: node2vec
#     feature_concat_embedding = False  # True / False, new embeddings with original embeddings / features together
#     use_shuffle = True
#     neg_pos_ratio = 1

#     # 2. generator
#     tr_ge_divide_ratio = 0.5

#     # x. common
#     work_dir = "./"
#     dataset_dir = work_dir + "datasets/"
#     out_dir = work_dir + "output/"


# class Profile(Context):
#     C = Context

#     dataset = os.environ["dataset"]
#     if os.environ["multi_class"] == "T":
#         multi_class = True
#     else:
#         multi_class = False



# P = Profile


# class LPLoader:
#     def __init__(self):
#         self.graph_data = GraphData()
#         self.graph_data.relations.get(GraphData.NO_EDGE)
#         if not P.multi_class:
#             Logger.log("add type HAS_EDGE to relations.")
#             self.graph_data.relations.get(GraphData.HAS_EDGE)

#     def load_data(self):
#         Logger.log(P.dataset)
#         if P.dataset == "dblp":
#             graph = self.dblp()
#         if P.dataset == "wikidata1k":
#             graph = self.wiki(1000)
#         if P.dataset == "wikidata5k":
#             graph = self.wiki(5000)
#         if P.dataset == "wikidata10k":
#             graph = self.wiki(10000)
#         if P.dataset == "ppi":
#             graph = self.ppi(3000)
#         if P.dataset == "twitter":
#             graph = self.twitter(100)
#         if P.dataset == "blogcatalog":
#             graph = self.blogcatalog()

#         # add embedding (optional)
#         if P.embedding_method != 0:
#             self.add_embedding(graph)

#         # convert to target format
#         n_node_class = 3  # at least 3
#         x = []
#         y = []
#         edge_index = [[], []]
#         for i in graph:
#             x.append(graph[i]["feature"])
#             if "label" in graph[i]:
#                 y.append(graph[i]["label"])
#                 n_node_class = max(y) + 1
#             else:
#                 y.append(i % n_node_class)  # random add class for nodes
#             if len(graph[i]["edges"]) == 0:  # for a ndoe without edge, add self loop
#                 edge_index[0].append(i)
#                 edge_index[1].append(i)
#             for e in graph[i]["edges"]:
#                 edge_index[0].append(i)
#                 edge_index[1].append(e)

#         train_mask = [False] * len(y)
#         val_mask = [False] * len(y)
#         test_mask = [False] * len(y)

#         for i in range(len(y)):
#             if i < int(P.tr_ge_divide_ratio * len(y)):
#                 train_mask[i] = True
#             elif i <= int(P.tr_ge_divide_ratio * len(y)) + n_node_class - 1:
#                 val_mask[i] = True
#                 y[i] = int(P.tr_ge_divide_ratio * len(y)) + n_node_class - i - 1
#             elif i > int(P.tr_ge_divide_ratio * len(y)) + n_node_class:
#                 test_mask[i] = True

#         data = Data(
#             x=torch.tensor(np.array(x), dtype=torch.float32), 
#             edge_index=torch.tensor(np.array(edge_index), dtype=torch.int64), 
#             y=torch.tensor(np.array(y), dtype=torch.int64),
#             train_mask=torch.tensor(np.array(train_mask).reshape(-1, 1)),
#             val_mask=torch.tensor(np.array(val_mask).reshape(-1, 1)),
#             test_mask=torch.tensor(np.array(test_mask).reshape(-1, 1)),
#         )

#         return graph, data

#     def add_embedding(self, graph, undirected=False):
#         # Make graph
#         nodes = list(graph)
#         edges = []
#         embeddings = None
#         for node in graph:
#             for edge in graph[node]["edges"]:
#                 if edge in graph:
#                     edges.append((int(node), int(edge)))
#         if P.embedding_method == 1:  # node2vec
#             nx_graph = nx.DiGraph()
#             nx_graph.add_nodes_from(nodes)
#             nx_graph.add_edges_from(edges)
#             # undirected graph
#             if undirected:
#                 nx_graph.to_undirected()
#             node2vec = Node2Vec(
#                 nx_graph, dimensions=256, walk_length=30, num_walks=200, workers=4
#             )
#             model = node2vec.fit(window=10, min_count=1, batch_words=4)
#             embeddings = model.wv

#         for e in graph:
#             if P.feature_concat_embedding:
#                 if P.embedding_method == 1:  # node2vec
#                     graph[e]["feature"] = np.concatenate((
#                         graph[e]["feature"],
#                         np.array(list(embeddings[str(e)]), dtype=float)
#                     ), axis=0)
#             else:
#                 if P.embedding_method == 1:  # node2vec
#                     graph[e]["feature"] = np.array(list(embeddings[str(e)]), dtype=float)

#     def dblp(self, redownload=False):
#         if redownload:
#             import torch_geometric.transforms as T
#             from torch_geometric.datasets import AttributedGraphDataset, CitationFull
#             dataset = CitationFull(P.dataset_dir + "dblp/", 'DBLP', transform=T.NormalizeFeatures())
#             data = dataset[0]
#             x = data.x.cpu().detach().numpy()
#             edge_index = data.edge_index.cpu().detach().numpy()
#             y = data.y.cpu().detach().numpy()

#             with open(P.dataset_dir + "dblp/dblp.pkl", 'wb') as file:
#                 pickle.dump([x, edge_index, y], file)

#         with open(P.dataset_dir + "dblp/dblp.pkl", 'rb') as file:
#             x, edge_index, y = pickle.load(file)

#         graph = dict()
#         for i in range(x.shape[0]):
#             graph[i] = dict()
#             graph[i]['edges'] = dict()
#             graph[i]['feature'] = x[i]
#             graph[i]["label"] = y[i]
#         for i in range(edge_index.shape[1]):
#             from_node = edge_index[0][i]
#             to_node = edge_index[1][i]
#             graph[from_node]['edges'][to_node] = 1

#         return graph

#     def wiki(self, num):
#         wiki2vec = Wikipedia2Vec.load(P.dataset_dir + "wikidata/enwiki_20180420_win10_500d.pkl")
#         with open(P.dataset_dir + "wikidata/wikidata-20150921-250k.json", "r") as raw_data:
#             entity_num = 0
#             edge_num = 0
#             for line in raw_data:
#                 # constraint of max entity
#                 if entity_num >= num:
#                     break

#                 # filter
#                 if line.startswith("["):
#                     continue
#                 if line.startswith("]"):
#                     continue

#                 # convert to json format
#                 json_raw = json.loads(line.strip().strip(","))

#                 # filter properties
#                 if str(json_raw["id"]).startswith("P"):
#                     continue

#                 # entity_dict is { 'edges' : {}, 'feature' : {}}
#                 entity_dict = dict()

#                 # edges is like { 1 : 3, 2 : 9 } or { 1 : 0, 2 : 1 }
#                 edges = dict()
#                 if "claims" not in json_raw:
#                     continue
#                 for claim in json_raw["claims"]:
#                     for link in json_raw["claims"][claim]:
#                         if str(link["type"]).startswith("statement"):
#                             if "datavalue" not in link["mainsnak"]:
#                                 continue
#                             if str(link["mainsnak"]["datavalue"]["type"]).startswith("wikibase-entityid"):
#                                 to_node_ind = int(link["mainsnak"]["datavalue"]["value"]["numeric-id"])
#                                 if P.multi_class:
#                                     edges[to_node_ind] = self.graph_data.relations.get(int(claim[1:]))
#                                 else:
#                                     edges[to_node_ind] = self.graph_data.relations.get(GraphData.HAS_EDGE)
#                 entity_dict["edges"] = edges
#                 edge_num += len(edges)

#                 # feature search
#                 try:   
#                     label = str(json_raw["labels"]["en"]["value"]).title()
#                     label = label.replace("The", "the")
#                     label = label.replace("And", "and")
#                     label = label.replace("Of", "of")
#                     # shape of entity_dict["feature"] is (500, )
#                     entity_dict["feature"] = np.array(wiki2vec.get_entity_vector(label).tolist(), dtype=float)
#                 except KeyError:
#                     # some entity feature can't be found or has not english label
#                     continue

#                 # add entity_dic to nodes
#                 node_ind = self.graph_data.node_map.get(int(json_raw["id"][1:]))
#                 self.graph_data.graph[node_ind] = entity_dict

#                 entity_num += 1

#             # map the index
#             map = dict()
#             new_graph = dict()
#             for i, n in enumerate(self.graph_data.graph):
#                 map[n] = i
#             for n in self.graph_data.graph:
#                 new_graph[map[n]] = dict()
#                 new_graph[map[n]]["feature"] = self.graph_data.graph[n]["feature"]
#                 new_graph[map[n]]["edges"] = dict()
#                 for e in self.graph_data.graph[n]["edges"]:
#                     if e in map:
#                         new_graph[map[n]]["edges"][map[e]] = self.graph_data.graph[n]["edges"][e]
            
#             return new_graph

#     def twitter(self, num=100):
#         feature_dicts = dict()
#         edge_dicts = dict()
#         entity_num = 0
#         processed_egonet = []
#         for file in os.listdir(P.dataset_dir + "twitter/"):
#             if entity_num >= num:
#                 break

#             node = self.graph_data.node_map.get(int(str(file).split(".")[0]))
#             if node not in processed_egonet:
#                 processed_egonet.append(node)
#                 if node not in feature_dicts:
#                     feature_dicts[node] = dict()
#                     entity_num += 1
#                     edge_dicts[node] = dict()
#                 keys = list()
#                 with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".featnames", "r") as f:
#                     for line in f:
#                         keys.append(line.rstrip().upper().split(" ")[1])
#                 values = list()
#                 with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".egofeat", "r") as f:
#                     for line in f:
#                         values = line.split(" ")
#                 for key, value in zip(keys, values):
#                     feature_dicts[node][key] = int(value)
#                 with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".feat", "r") as f:
#                     for line in f:
#                         splits = line.split(" ")
#                         sub_node = self.graph_data.node_map.get(int(splits[0]))
#                         if sub_node not in feature_dicts:
#                             feature_dicts[sub_node] = dict()
#                             entity_num += 1
#                             edge_dicts[sub_node] = dict()
#                         if sub_node not in edge_dicts[node]:  # all nodes in the ego network connect to the ego user
#                             edge_dicts[node][sub_node] = self.graph_data.relations.get(GraphData.HAS_EDGE)
#                         sub_values = splits[1:]
#                         for key, value in zip(keys, sub_values):
#                             feature_dicts[sub_node][key] = int(value)

#                 with open(P.dataset_dir + "twitter/" + str(self.graph_data.node_map.inverse(node)) + ".edges", "r") as f:
#                     for line in f:
#                         splits = line.split(" ")
#                         from_node = self.graph_data.node_map.get(int(splits[0]))
#                         to_node = self.graph_data.node_map.get(int(splits[1]))
#                         if to_node not in edge_dicts[from_node]:
#                             edge_dicts[from_node][to_node] = self.graph_data.relations.get(GraphData.HAS_EDGE)

#         # get all features appeared in feature_dicts
#         all_feature = dict()
#         for node in feature_dicts:
#             for key in feature_dicts[node]:
#                 if key not in all_feature:
#                     if feature_dicts[node][key] > 0:
#                         all_feature[key] = 1
#                 else:
#                     all_feature[key] += 1

#         # select the most common features (top x)
#         top_feature = dict()
#         top_feature_value = list(all_feature.values())
#         top_feature_value.sort(reverse=True)

#         # debug: see the dimension
#         # print(top_feature_value)  # see the top feature value to judge the dim needed
#         # exit(0)

#         if P.dataset == "twitter":
#             top_feature_value = top_feature_value[:P.twitter_feat_dim]
#         for feat in all_feature:
#             if all_feature[feat] in top_feature_value:
#                 top_feature[feat] = True
#         Logger.log("feature dimension is {}".format(len(top_feature)))

#         # generate feature vec for all nodes
#         all_feature_dicts = dict()
#         for node in feature_dicts:
#             feature_vec = list()
#             for key in top_feature:
#                 if key not in feature_dicts[node]:
#                     feature_vec.append(0)
#                 elif feature_dicts[node][key] == 0:
#                     feature_vec.append(0)
#                 elif feature_dicts[node][key] == 1:
#                     feature_vec.append(1)
#             all_feature_dicts[node] = feature_vec

#         # make data
#         for node in feature_dicts:
#             self.graph_data.graph[node] = dict()
#             # edges: edges is like { 1 : 0, 2 : 1 }
#             self.graph_data.graph[node]["edges"] = edge_dicts[node]
#             self.graph_data.graph[node]["feature"] = np.array(all_feature_dicts[node], dtype=float)

#         Logger.log("use {} ego nets".format(len(processed_egonet)))

#         return self.graph_data.graph

#     def ppi(self, num=1000):
#         with open(P.dataset_dir + "ppi/" + "ppi-class_map.json", "r") as class_data:
#             protein_class = json.load(class_data)
#         protein_feats = np.load(P.dataset_dir + "ppi/" + "ppi-feats.npy")
#         with open(P.dataset_dir + "ppi/" + "ppi-G.json", "r") as graph_data:
#             ppi_graph = json.load(graph_data)

#         entity_num = 0
#         edge_num = 0
#         # entity is { 'edges' : {}, 'feature' : {}}
#         for i in range(len(ppi_graph["nodes"])):
#             if Funcs.rand_prob() > 0.1:
#                 continue
#             # constraint of max entity
#             if entity_num >= num:
#                 break

#             node_id = self.graph_data.node_map.get(ppi_graph["nodes"][i]["id"])
#             if node_id not in self.graph_data.graph:
#                 self.graph_data.graph[node_id] = dict()
#                 # edges: edges is like { 1 : 0, 2 : 1 }
#                 self.graph_data.graph[node_id]["edges"] = dict()
#                 # feature: feature is like {0.1, 0.5, 0.8}
#                 self.graph_data.graph[node_id]["feature"] = np.array(
#                     list(protein_class[str(self.graph_data.node_map.inverse(node_id))]) +
#                     list(protein_feats[self.graph_data.node_map.inverse(node_id)]), dtype=float)
#                 entity_num = entity_num + 1

#         for link in ppi_graph["links"]:
#             # filtered edges with to_node not in self.graph_data.nodes
#             from_node_ind = self.graph_data.node_map.get(link["source"])
#             to_node_ind = self.graph_data.node_map.get(link["target"])
#             if from_node_ind in self.graph_data.graph and to_node_ind in self.graph_data.graph:
#                 if to_node_ind not in self.graph_data.graph[from_node_ind]["edges"]:
#                     self.graph_data.graph[from_node_ind]["edges"][to_node_ind] = self.graph_data.relations.get(GraphData.HAS_EDGE)
#                     edge_num += 1

#         return self.graph_data.graph

#     def blogcatalog(self, redownload=False):
#         if redownload:
#             import torch_geometric.transforms as T
#             from torch_geometric.datasets import AttributedGraphDataset, CitationFull
#             dataset = AttributedGraphDataset(P.dataset_dir + 'blogcatalog/', 'blogcatalog', transform=T.NormalizeFeatures())
#             data = dataset[0]
#             x = data.x.cpu().detach().numpy()  # (5196, 8189)
#             edge_index = data.edge_index.cpu().detach().numpy()  # (2, 343486)
#             y = data.y.cpu().detach().numpy()  # (5196,), 6 classes

#             with open(P.dataset_dir + "blogcatalog/blogcatalog.pkl", 'wb') as file:
#                 pickle.dump([x, edge_index, y], file)

#         with open(P.dataset_dir + "blogcatalog/blogcatalog.pkl", 'rb') as file:
#             x, edge_index, y = pickle.load(file)

#         graph = dict()
#         for i in range(x.shape[0]):
#             graph[i] = dict()
#             graph[i]['edges'] = dict()
#             graph[i]['feature'] = x[i]
#             graph[i]["label"] = y[i]
#         for i in range(edge_index.shape[1]):
#             from_node = edge_index[0][i]
#             to_node = edge_index[1][i]
#             graph[from_node]['edges'][to_node] = 1

#         return graph


# class LPEval():
#     @staticmethod
#     def eval(graph, emb, multi_class=P.multi_class, use_shuffle=P.use_shuffle, neg_pos_ratio=P.neg_pos_ratio, split_ratio=P.tr_ge_divide_ratio):
#         # feeder
#         positive_sample = []
#         negative_sample = []

#         for node_id in graph:
#             for edge in graph[node_id]["edges"]:
#                 if edge in graph:
#                     positive_sample.append((node_id, edge, graph[node_id]["edges"][edge]))

#         if not multi_class:
#             node_num = len(graph)
#             target_neg_num = math.ceil(len(positive_sample) * neg_pos_ratio)
#             if target_neg_num < 1:
#                 target_neg_num = 1
#             while len(negative_sample) < target_neg_num:
#                 rand_from_node = Funcs.rand_int(0, node_num - 1)
#                 rand_to_node = Funcs.rand_int(0, node_num - 1)
#                 from_node = list(graph.keys())[rand_from_node]
#                 to_node = list(graph.keys())[rand_to_node]
#                 if (from_node, to_node) not in positive_sample:
#                     negative_sample.append((from_node, to_node, 0))

#         # shuffle train and test
#         if use_shuffle:
#             Funcs.shuffle_list(positive_sample)
#             Funcs.shuffle_list(negative_sample)

#         train_pos = positive_sample[:int(len(positive_sample) * split_ratio)]
#         test_pos = positive_sample[int(len(positive_sample) * split_ratio):]

#         train_neg = negative_sample[:int(len(negative_sample) * split_ratio)]
#         test_neg = negative_sample[int(len(negative_sample) * split_ratio):]

#         train_edges, train_edges_false, test_edges, test_edges_false = train_pos, train_neg, test_pos, test_neg
        
#         # compute embeddings
#         emb_mappings = dict()
#         for i in range(len(emb)):
#             emb_mappings[i] = emb[i]

#         # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
#         def get_edge_embeddings(edge_list):
#             embs = []
#             for edge in edge_list:
#                 node1 = edge[0]
#                 node2 = edge[1]
#                 emb1 = emb_mappings[node1]
#                 emb2 = emb_mappings[node2]
#                 edge_emb = np.multiply(emb1, emb2)
#                 embs.append(edge_emb)
#             embs = np.array(embs)
#             return embs

#         # Train-set edge embeddings
#         pos_train_edge_embs = get_edge_embeddings(train_edges)
#         neg_train_edge_embs = get_edge_embeddings(train_edges_false)
#         if multi_class:
#             train_edge_embs = pos_train_edge_embs
#             train_edge_labels = np.array([e[2] for e in train_edges])
#         else:
#             train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
#             train_edge_labels = np.array([e[2] for e in (train_edges + train_edges_false)])

#         # Test-set edge embeddings, labels
#         pos_test_edge_embs = get_edge_embeddings(test_edges)
#         neg_test_edge_embs = get_edge_embeddings(test_edges_false)
#         if multi_class:
#             test_edge_embs = pos_test_edge_embs
#             test_edge_labels = np.array([e[2] for e in test_edges])
#         else:
#             test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
#             test_edge_labels = np.array([e[2] for e in (test_edges + test_edges_false)])

#         # Train logistic regression classifier on train-set edge embeddings
#         if multi_class:
#             edge_classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
#         else:
#             edge_classifier = LogisticRegression(random_state=0)
#         edge_classifier.fit(np.array(train_edge_embs), np.array(train_edge_labels))

#         # Predicted edge scores: probability of being of class "1" (real edge)
#         test_preds = edge_classifier.predict(test_edge_embs)
#         test_probs = edge_classifier.predict_proba(test_edge_embs)

#         # record result
#         predicted = list()
#         ground_truth = list()
#         score = list()
#         for i in range(len(test_edge_labels)):
#             # print("--- {} - {} ---".format(test_preds[i], test_edge_labels[i]))
#             predicted.append(test_preds[i])
#             score.append(test_probs[i][1])
#             ground_truth.append(test_edge_labels[i])

#         if len(predicted) == 0:
#             Logger.log("predicted value is empty.")
#             return

#         if multi_class:
#             # accuracy
#             accuracy = skm.accuracy_score(ground_truth, predicted)

#             labels = set()
#             for e in ground_truth:
#                 labels.add(e)

#             # Micro-F1
#             micro_f1 = skm.f1_score(ground_truth, predicted, labels=list(labels), average="micro")

#             # Macro-F1
#             macro_f1 = skm.f1_score(ground_truth, predicted, labels=list(labels), average="macro")

#             Logger.log("Acc: {:.4f} Micro-F1: {:.4f} Macro-F1: {:.4f}".format(accuracy, micro_f1, macro_f1))
#         else:
#             # auc
#             auc = skm.roc_auc_score(ground_truth, score)

#             # accuracy
#             accuracy = skm.accuracy_score(ground_truth, predicted)

#             # recall
#             recall = skm.recall_score(ground_truth, predicted)

#             # precision
#             precision = skm.precision_score(ground_truth, predicted)

#             # F1
#             f1 = skm.f1_score(ground_truth, predicted)

#             # AUPR
#             pr, re, _ = skm.precision_recall_curve(ground_truth, score)
#             aupr = skm.auc(re, pr)

#             # AP
#             ap = skm.average_precision_score(ground_truth, score)

#             Logger.log("Acc: {:.4f} AUC: {:.4f} Pr: {:.4f} Re: {:.4f} F1: {:.4f} AUPR: {:.4f} AP: {:.4f}".format(accuracy, auc, precision, recall, f1, aupr, ap))





class Logger:
    path = f".log"
    color = None
    style = None

    @staticmethod
    def log(msg, color=None, style=None, new_line=True):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if len(logger.handlers) == 0:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        elif not isinstance(logger.handlers[0], logging.StreamHandler) or len(logger.handlers) > 1:
            while len(logger.handlers) > 0:
                logger.removeHandler(logger.handlers[0])  # empty exist handler
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        handler = logger.handlers[0]
        if not new_line:
            handler.terminator = ""
        formatter = logging.Formatter(
            datefmt="",
            fmt="%(message)s"
        )
        handler.setFormatter(formatter)

        normal = "\033[0m"
        black = "\033[30m"
        red = "\033[31m"
        green = "\033[32m"
        yellow = "\033[33m"
        blue = "\033[34m"
        purple = "\033[35m"
        cyan = "\033[36m"
        grey = "\033[37m"

        bold = "\033[1m"
        italic = "\033[3m"
        underline = "\033[4m"
        invert = "\033[7m"
        remove = "\033[9m"

        msg = str(msg)
        raw_msg = msg

        if color is None:
            color = Logger.color

        if color == "blue":
            msg = blue + msg + normal
        elif color == "yellow":
            msg = yellow + msg + normal
        elif color == "black":
            msg = black + msg + normal
        elif color == "cyan":
            msg = cyan + msg + normal
        elif color == "grey":
            msg = grey + msg + normal
        elif color == "red":
            msg = red + msg + normal
        elif color == "green":
            msg = green + msg + normal
        elif color == "purple":
            msg = purple + msg + normal
        else:
            pass

        if style is None:
            style = Logger.style

        if style == "bold":
            msg = bold + msg + normal
        elif style == "underline":
            msg = underline + msg + normal
        elif style == "italic":
            msg = italic + msg + normal
        elif style == "invert":
            msg = invert + msg + normal
        elif style == "remove":
            msg = remove + msg + normal
        else:
            pass

        title_str = Logger.make_msg_title()
        title = cyan + title_str[0][0: 6] + yellow + title_str[0][6:] + blue + title_str[1] + green + title_str[2] + normal + " "
        logger.info(title + msg)
        if Logger.path is not None:
            Logger.write_log(msg=f"{''.join(title_str)} {raw_msg}", path=Logger.path)

    @staticmethod
    def make_msg_title():
        """
        year+month+day+hour+minute+second+CPU+Memory+GPU+GMem
        :return:
        """
        cpu_utilization, mem_used, gpu_info_list, all_gpu_utilization, all_gpu_mem_used = \
            Logger.get_hardware_info()
        date = Logger.get_date()
        title = "{}|{:>3}{:>2}|{:>3}{:>2}".format(
            date, int(cpu_utilization), int(mem_used), int(all_gpu_utilization), int(all_gpu_mem_used)
        )

        return title.split("|")

    @staticmethod
    def get_date():
        return datetime.datetime.fromtimestamp(
            int(time.time()),
            pytz.timezone("Asia/Shanghai")
        ).strftime("%y%m%d%H%M%S")

    @staticmethod
    def get_hardware_info():
        # output system info
        try:
            with open("/proc/meminfo") as mem_file:
                total = int(mem_file.readline().split()[1])  # total memory
                free = int(mem_file.readline().split()[1])  # free memory
            mem_used = ((total - free) / total) * 100
            cpu_utilization = psutil.cpu_percent(0)
        except FileNotFoundError:
            cpu_utilization, mem_used = -1, -1

        try:
            import pynvml
            pynvml.nvmlInit()

            all_gpu_utilization = 0
            all_gpu_mem_used = 0
            gpu_info_list = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_utilization_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = (gpu_mem_info.used / gpu_mem_info.total) * 100
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                gpu_fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                gpu_power_max = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000  # milli-watts / 1000
                gpu_power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

                gpu_info_list.append([i, gpu_utilization_rate.gpu, gpu_mem_used, gpu_temp,
                                      gpu_fan, gpu_power_usage, gpu_power_max])
                all_gpu_utilization += gpu_utilization_rate.gpu
                all_gpu_mem_used += gpu_mem_used

            all_gpu_utilization = all_gpu_utilization / pynvml.nvmlDeviceGetCount()
            all_gpu_mem_used = all_gpu_mem_used / pynvml.nvmlDeviceGetCount()

            return cpu_utilization, mem_used, gpu_info_list, all_gpu_utilization, all_gpu_mem_used
        except Exception:
            return cpu_utilization, mem_used, [], -1, -1

    @staticmethod
    def write_log(msg, path):
        file_name = str(path).split("/")[-1]
        directory = str(path)[0:(len(path) - len(file_name))]
        IO.make_dir(directory)
        with open(path, "a+") as log_file:
            log_file.write(msg + "\n")



class IO:
    @staticmethod
    def write_disk_dump(path, target_object):
        file_name = str(path).split("/")[-1]
        directory = str(path)[0:(len(path) - len(file_name))]
        IO.make_dir(directory)
        with open(path, "wb") as object_persistent:
            pickle.dump(target_object, object_persistent)

    @staticmethod
    def read_disk_dump(path):
        with open(path, "rb") as object_persistent:
            restore = pickle.load(object_persistent)
        return restore

    @staticmethod
    def read_file(path):
        content = []
        with open(path, "r") as file:
            for line in file:
                content.append(line.replace('\n', ''))
        return content

    @staticmethod
    def stick_read_disk_dump(path):
        while True:
            try:
                return IO.read_disk_dump(path)
            except Exception:
                time.sleep(1)

    @staticmethod
    def make_dir(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy(src, dst):
        shutil.copytree(src, dst)

    @staticmethod
    def delete_dir(directory):
        shutil.rmtree(directory, True)

    @staticmethod
    def renew_dir(directory):
        IO.delete_dir(directory)
        IO.make_dir(directory)

    @staticmethod
    def delete_file(path):
        if Path(path).exists():
            os.remove(path)

    @staticmethod
    def move_file(src, dst):
        shutil.move(src, dst)

    @staticmethod
    def list_dir(directory):
        files_and_dirs = os.listdir(directory)
        return files_and_dirs

    @staticmethod
    def cascaded_list_dir(directory, ext_filter=None):
        files = []
        dirs = []
        # r = root, d = directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                if ext_filter is not None:
                    if ext_filter in file:
                        files.append(os.path.join(r, file))
                else:
                    files.append(os.path.join(r, file))
            for di in d:
                dirs.append(di)
        return files, dirs