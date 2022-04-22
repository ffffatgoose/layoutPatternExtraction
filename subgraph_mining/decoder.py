import argparse
import csv
import pdb
from itertools import combinations
import time
import os

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from common import data
from common import models
from common import utils
from common import combined_syn

from common.own_dataset_decoder import OwnDataset

from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from subgraph_mining.search_agents import GreedySearchAgent, MCTSSearchAgent

import matplotlib.pyplot as plt

import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA

corrected_edge_dict = {2: "upside", 3: "upper left", 4: "left",
              5: "lower left", 6: "outside"}

def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    # PATTERN 1
    pattern = generator.generate(size=10)
    # PATTERN 2
    #pattern = nx.star_graph(9)
    # PATTERN 3
    #pattern = nx.complete_graph(10)
    # PATTERN 4
    #pattern = nx.Graph()
    #pattern.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    #    (6, 7), (7, 2), (7, 8), (8, 9), (9, 10), (10, 6)])
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs

def pattern_growth(dataset, task, args):
    # init model
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = models.BaselineMLP(1, args.hidden_dim, args)
    else:
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    if task == "graph-labeled":
        dataset, labels = dataset

    # load data
    neighs_pyg, neighs = [], []
    print(len(dataset), "graphs")
    print("search strategy:", args.search_strategy)
    if task == "graph-labeled": print("using label 0")
    graphs = []
    # pdb.set_trace()
    for i, graph in enumerate(dataset):
        if task == "graph-labeled" and labels[i] != 0: continue
        if task == "graph-truncate" and i >= 1000: break
        if not type(graph) == nx.Graph:
            # graph = pyg_utils.to_networkx(graph).to_undirected()

            # directed graph transform
            graph = pyg_utils.to_networkx(graph, node_attrs = ["x"], edge_attrs = ["edge_attr"])

        graphs.append(graph)

    args.use_whole_graphs = False
    print("whole graphs?",args.use_whole_graphs)

    if args.use_whole_graphs:
        start_time = time.time()
        neighs = graphs
    else:
        anchors = []
        if args.sample_method == "radial":
            for i, graph in enumerate(graphs):
                print(i)
                for j, node in enumerate(graph.nodes):
                    if len(dataset) <= 10 and j % 100 == 0: print(i, j)
                    if args.use_whole_graphs:
                        neigh = graph.nodes
                    else:
                        neigh = list(nx.single_source_shortest_path_length(graph,
                            node, cutoff=args.radius).keys())
                        if args.subgraph_sample_size != 0:
                            neigh = random.sample(neigh, min(len(neigh),
                                args.subgraph_sample_size))
                    if len(neigh) > 1:
                        neigh = graph.subgraph(neigh)
                        if args.subgraph_sample_size != 0:
                            neigh = neigh.subgraph(max(
                                nx.connected_components(neigh), key=len))
                        neigh = nx.convert_node_labels_to_integers(neigh)
                        neigh.add_edge(0, 0)
                        neighs.append(neigh)
        elif args.sample_method == "tree":
            start_time = time.time()
            # sample n_neighborhoods graph to compare/judge if subgraph or not
            for j in tqdm(range(args.n_neighborhoods)):
                graph, neigh = utils.sample_neigh(graphs,
                    random.randint(args.min_neighborhood_size,
                        args.max_neighborhood_size))
                neigh = graph.subgraph(neigh)
                neigh = nx.convert_node_labels_to_integers(neigh)
                # neigh.add_edge(0, 0) # already added in data
                neighs.append(neigh)
                # print("node anchored:",args.node_anchored) false
                if args.node_anchored:
                    neigh.add_edge(0, 0)
                    anchors.append(0)   # after converting labels, 0 will be anchor

    embs = []
    if len(neighs) % args.batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in range(len(neighs) // args.batch_size):
        #top = min(len(neighs), (i+1)*args.batch_size)
        top = (i+1)*args.batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(neighs[i*args.batch_size:top],
                anchors=anchors if args.node_anchored else None, mode='Decoder')
            emb = model.emb_model(batch)
            emb = emb.to(torch.device("cpu"))

        embs.append(emb)

    print("Finish emb")

    if args.analyze:
        embs_np = torch.stack(embs).numpy()
        plt.scatter(embs_np[:,0], embs_np[:,1], label="node neighborhood")

    if args.search_strategy == "mcts":
        assert args.method_type == "order"
        agent = MCTSSearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, out_batch_size=args.out_batch_size)
    elif args.search_strategy == "greedy":
        agent = GreedySearchAgent(args.min_pattern_size, args.max_pattern_size,
            model, graphs, embs, node_anchored=args.node_anchored,
            analyze=args.analyze, model_type=args.method_type,
            out_batch_size=args.out_batch_size)
    out_graphs = agent.run_search(args.n_trials)
    print(time.time() - start_time, "TOTAL TIME")
    x = int(time.time() - start_time)
    print(x // 60, "mins", x % 60, "secs")

    # visualize out patterns
    count_by_size = defaultdict(int)

    # some file data to better visualization
    # dataset204 config
    # with open(r"C:\fatgoose\MSRA\codeForSync\buildGraphAndStatic\subgroup-data\Origin_shapeName_idx_dicttype2-layoutGMN-decoder-withgroup-20333","r",encoding='utf-8') as f:
    #     shapeNameOrigin_idx_dict = eval(f.readline())
    # shapeIdx_OriginName_dict = dict(zip(shapeNameOrigin_idx_dict.values(), shapeNameOrigin_idx_dict.keys()))
    #
    # # graph_dictreach-abso-10edge-decoder-t2 graphIdx_file_dict
    # with open(r"C:\fatgoose\MSRA\codeForSync\buildGraphAndStatic\subgroup-data\graph_dicttype2-layoutGMN-decoder-withgroup-20333","r",encoding='utf-8') as f:
    #     graphIdx_file_dict = eval(f.readline())

    # dataset205 config
    with open(r"./dataset/decoder-visualization/Origin_shapeName_idx_dicttype2-layoutGMN-decoder-withgroup-205","r",encoding='utf-8') as f:
        shapeNameOrigin_idx_dict = eval(f.readline())
    shapeIdx_OriginName_dict = dict(zip(shapeNameOrigin_idx_dict.values(), shapeNameOrigin_idx_dict.keys()))

    with open(r"./dataset/decoder-visualization/graph_dicttype2-layoutGMN-decoder-withgroup-205","r",encoding='utf-8') as f:
        graphIdx_file_dict = eval(f.readline())


    for pattern in out_graphs:
        if args.node_anchored:
            colors = ["red"] + ["blue"]*(len(pattern)-1)
            nx.draw(pattern, node_color=colors, with_labels=True)
        else:
            # delete one path between two nodes,for display two edges together can not see edge clearly
            delete_edge_set = set()
            for edge_i in pattern.edges:
                if pattern.has_edge(edge_i[1],edge_i[0]):
                    if pattern[edge_i[0]][edge_i[1]]['edge_attr'][-1] >= 7:
                        delete_edge_set.add((edge_i[0],edge_i[1]))
                    else:
                        delete_edge_set.add((edge_i[1], edge_i[0]))
            for node_i,node_j in delete_edge_set:
                print(node_i,node_j)
                pattern.remove_edge(node_i,node_j)

            pos = nx.spring_layout(pattern)
            nx.draw(pattern, pos)
            node_labels_idx = nx.get_node_attributes(pattern, 'x')

            node_labels_name = {key:shapeIdx_OriginName_dict[int(value[-2])] for key,value in node_labels_idx.items()}
            nx.draw_networkx_labels(pattern, pos, labels=node_labels_name,font_size=10)
            edge_labels_idx = nx.get_edge_attributes(pattern, 'edge_attr')
            edge_labels_name = {key:corrected_edge_dict[int(value[-1])] for key,value in edge_labels_idx.items()}
            nx.draw_networkx_edge_labels(pattern,pos,edge_labels=edge_labels_name,font_color="blue")

            for key, value in node_labels_idx.items():
                tmp_graph_idx = int(value[-1])
                break
            # plt.text(0,0,str(graphIdx_file_dict[tmp_graph_idx-1]),color='r') # dataset 204,205 from same layout

            # plt.show()


        plt.savefig("plots/cluster/{}-{}-201-205-test1.png".format(len(pattern),
                                                     count_by_size[len(pattern)]))
        print("Saving plots/cluster/{}-{}-201-205-test1.png".format(len(pattern),
            count_by_size[len(pattern)]))

        plt.close()
        count_by_size[len(pattern)] += 1

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(args.out_path, "wb") as f:
        pickle.dump(out_graphs, f)

def main():
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()
    #args.dataset = "enzymes"

    print("Using dataset {}".format(args.dataset))
    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
        task = 'graph'
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
        task = 'graph'
    elif args.dataset == 'dblp':
        dataset = TUDataset(root='/tmp/dblp', name='DBLP_v1')
        task = 'graph-truncate'
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
        task = 'graph'
    elif args.dataset.startswith('roadnet-'):
        graph = nx.Graph()
        with open("data/{}.txt".format(args.dataset), "r") as f:
            for row in f:
                if not row.startswith("#"):
                    a, b = row.split("\t")
                    graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif args.dataset == "ppi":
        dataset = PPI(root="/tmp/PPI")
        task = 'graph'
    elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {"diseasome": "bio-diseasome.mtx",
            "usroads": "road-usroads.mtx",
            "mn-roads": "mn-roads.mtx",
            "infect": "infect-dublin.edges"}
        graph = nx.Graph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
        task = 'graph'
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = make_plant_dataset(size)
        task = 'graph'
    elif args.dataset == "arxiv":
        dataset = PygNodePropPredDataset(name = "ogbn-arxiv")
        task = "graph"
    elif args.dataset == 'vaulttype2GMN04posA':#vaultnoSampleDecoderDelG
        dataset = OwnDataset(root='./dataset/vaulttype2GMN04posA', name='vaulttype2GMN04posA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"
    elif args.dataset == 'vaulttype2GMN010posA':#VAULT DATASET -WITHOUT GROUP
        dataset = OwnDataset(root='./dataset/vaulttype2GMN010posA', name='vaulttype2GMN010posA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"
    elif args.dataset == 'vaulttype2GMN020posA':# BETTER DATASET
        dataset = OwnDataset(root='./dataset/vaulttype2GMN020posA', name='vaulttype2GMN020posA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"
    elif args.dataset == 'vaulttype2GMN101withGroupDecoder':# BETTER DATASET
        dataset = OwnDataset(root='./dataset/vaulttype2GMN101WithGroupDecoderposA', name='vaulttype2GMN101WithGroupDecoderposA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"
    elif args.dataset == 'vaulttype2GMN201withGroupDecoder':# BETTER DATASET
        dataset = OwnDataset(root='./dataset/vaulttype2GMN201WithGroupDecoderposA', name='vaulttype2GMN201WithGroupDecoderposA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"
    elif args.dataset == 'vaulttype2GMN203withGroupDecoder':# BETTER DATASET x100 #vaulttype2GMN204DECODERposA
        dataset = OwnDataset(root='./dataset/vaulttype2GMN203WithGroupDecoderposA', name='vaulttype2GMN203WithGroupDecoderposA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"
    elif args.dataset == 'vaulttype2GMN204DECODERposA':# BETTER DATASET x100 #vaulttype2GMN204DECODERposA no group # ppt2 - 19
        dataset = OwnDataset(root='./dataset/vaulttype2GMN204DECODERposA', name='vaulttype2GMN204DECODERposA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"
    elif args.dataset == 'vaulttype2GMN205DECODERposA':# BETTER DATASET x100 no group # ppt101 - 6
        dataset = OwnDataset(root='./dataset/vaulttype2GMN205DECODERposA', name='vaulttype2GMN205DECODERposA', use_node_attr=True,
                             use_edge_attr=True)
        task = "graph"


    pattern_growth(dataset, task, args) 

if __name__ == '__main__':
    main()

