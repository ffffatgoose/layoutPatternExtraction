import os
import pdb
import pickle
import random

import networkx
from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset, Generator
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import queue
import scipy.stats as stats

from common import combined_syn
from common import feature_preprocess
from common import utils
from common.own_dataset_encoder import OwnDataset

def load_dataset(name):
    """ Load real-world datasets, available in PyTorch Geometric.

    Used as a helper for DiskDataSource.
    """
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    elif name == "proteins":
        dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name="COX2")
    elif name == "aids":
        dataset = TUDataset(root="/tmp/AIDS", name="AIDS")
    elif name == "reddit-binary":
        dataset = TUDataset(root="/tmp/REDDIT-BINARY", name="REDDIT-BINARY")
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name="IMDB-BINARY")
    elif name == "firstmm_db":
        dataset = TUDataset(root="/tmp/FIRSTMM_DB", name="FIRSTMM_DB")
    elif name == "dblp":
        dataset = TUDataset(root="/tmp/DBLP_v", name="DBLP_v")
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
    elif name == "qm9":
        dataset = QM9(root="/tmp/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]

    elif name == 'vaultGMN101Micro':
        dataset1 = OwnDataset(root='./dataset/vaultGMN101MicroPosA', name='vaultGMN101MicroPosA', use_node_attr=True,
                              use_edge_attr=True)
        dataset2 = OwnDataset(root='./dataset/vaultGMN101MicroPosB', name='vaultGMN101MicroPosB', use_node_attr=True,
                              use_edge_attr=True)
        dataset3 = OwnDataset(root='./dataset/vaultGMN101MicroNegA', name='vaultGMN101MicroNegA', use_node_attr=True,
                              use_edge_attr=True)
        dataset4 = OwnDataset(root='./dataset/vaultGMN101MicroNegB', name='vaultGMN101MicroNegB', use_node_attr=True,
                              use_edge_attr=True)
    elif name == 'vaultGMN201Micro':
        dataset1 = OwnDataset(root='./dataset/vaultGMN201MicroPosA', name='vaultGMN201MicroPosA', use_node_attr=True,
                              use_edge_attr=True)
        dataset2 = OwnDataset(root='./dataset/vaultGMN201MicroPosB', name='vaultGMN201MicroPosB', use_node_attr=True,
                              use_edge_attr=True)
        dataset3 = OwnDataset(root='./dataset/vaultGMN201MicroNegA', name='vaultGMN201MicroNegA', use_node_attr=True,
                              use_edge_attr=True)
        dataset4 = OwnDataset(root='./dataset/vaultGMN201MicroNegB', name='vaultGMN201MicroNegB', use_node_attr=True,
                              use_edge_attr=True)
    elif name == 'vaultGMN102Macro':
        dataset1 = OwnDataset(root='./dataset/vaultGMN102MacroPosA', name='vaultGMN102MacroPosA', use_node_attr=True,
                              use_edge_attr=True)
        dataset2 = OwnDataset(root='./dataset/vaultGMN102MacroPosB', name='vaultGMN102MacroPosB', use_node_attr=True,
                              use_edge_attr=True)
        dataset3 = OwnDataset(root='./dataset/vaultGMN102MacroNegA', name='vaultGMN102MacroNegA', use_node_attr=True,
                              use_edge_attr=True)
        dataset4 = OwnDataset(root='./dataset/vaultGMN102MacroNegB', name='vaultGMN102MacroNegB', use_node_attr=True,
                              use_edge_attr=True)
    elif name == 'vaultGMN202Macro':
        dataset1 = OwnDataset(root='./dataset/vaultGMN202MacroPosA', name='vaultGMN202MacroPosA', use_node_attr=True,
                              use_edge_attr=True)
        dataset2 = OwnDataset(root='./dataset/vaultGMN202MacroPosB', name='vaultGMN202MacroPosB', use_node_attr=True,
                              use_edge_attr=True)
        dataset3 = OwnDataset(root='./dataset/vaultGMN202MacroNegA', name='vaultGMN202MacroNegA', use_node_attr=True,
                              use_edge_attr=True)
        dataset4 = OwnDataset(root='./dataset/vaultGMN202MacroNegB', name='vaultGMN202MacroNegB', use_node_attr=True,
                              use_edge_attr=True)
    elif name == 'vaultGMN203MacroDIFF':
        dataset1 = OwnDataset(root='./dataset/vaultGMN203MacroDIFFPosA', name='vaultGMN203MacroDIFFPosA', use_node_attr=True,
                              use_edge_attr=True)
        dataset2 = OwnDataset(root='./dataset/vaultGMN203MacroDIFFPosB', name='vaultGMN203MacroDIFFPosB', use_node_attr=True,
                              use_edge_attr=True)
        dataset3 = OwnDataset(root='./dataset/vaultGMN203MacroDIFFNegA', name='vaultGMN203MacroDIFFNegA', use_node_attr=True,
                              use_edge_attr=True)
        dataset4 = OwnDataset(root='./dataset/vaultGMN203MacroDIFFNegB', name='vaultGMN203MacroDIFFNegB', use_node_attr=True,
                              use_edge_attr=True)
    elif name == 'vaulttype2GMN0301':#BETTER03 #vaultGMN101MicroPosB
        dataset1 = OwnDataset(root='./dataset/vaultGMN0301PosA', name='vaultGMN0301PosA', use_node_attr=True,
                              use_edge_attr=True)
        dataset2 = OwnDataset(root='./dataset/vaultGMN0301PosB', name='vaultGMN0301PosB', use_node_attr=True,
                              use_edge_attr=True)
        dataset3 = OwnDataset(root='./dataset/vaultGMN0301NegA', name='vaultGMN0301NegA', use_node_attr=True,
                              use_edge_attr=True)
        dataset4 = OwnDataset(root='./dataset/vaultGMN0301NegB', name='vaultGMN0301NegB', use_node_attr=True,
                              use_edge_attr=True)

    if task == "graph":
        train_len1 = int(0.5 * len(dataset1))
        train_len2 = int(0.5 * len(dataset3))
        # print("len dataset:",len(dataset1),len(dataset2),len(dataset3),len(dataset4))
        # print(train_len1,train_len2)
        train, test = [], []
        dataset = list(dataset1) + list(dataset2) + list(dataset3) + list(dataset4)

        # random.shuffle(dataset) #already
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset1)):
            if not type(graph) == nx.Graph:
                if has_name: del graph.name
                # change into undirected graphs
                # graph = pyg_utils.to_networkx(graph).to_undirected()

                # change into directed graphs
                #graph = pyg_utils.to_networkx(graph)

                # change into directed graphs with node_attr and edge_attr
                graph_pos_a = pyg_utils.to_networkx(graph, node_attrs = ["x"], edge_attrs = ["edge_attr"])
                graph_pos_b = pyg_utils.to_networkx(dataset2[i], node_attrs = ["x"], edge_attrs = ["edge_attr"])
                graph_neg_a = pyg_utils.to_networkx(dataset3[i], node_attrs=["x"], edge_attrs=["edge_attr"])
                graph_neg_b = pyg_utils.to_networkx(dataset4[i], node_attrs=["x"], edge_attrs=["edge_attr"])

            if i < train_len1:
                train.append(graph_pos_a)
                train.append(graph_pos_b)
                train.append(graph_neg_a)
                train.append(graph_neg_b)

            else:
                test.append(graph_pos_a)
                test.append(graph_pos_b)
                test.append(graph_neg_a)
                test.append(graph_neg_b)

    return train, test, task

class DataSource:
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError

class OTFSynDataSource(DataSource):
    """ On-the-fly generated synthetic data for training the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    with a pre-defined generator (see combined_syn.py).

    DeepSNAP transforms are used to generate the positive and negative examples.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        self.closed = False
        self.max_size = max_size
        self.min_size = min_size
        self.node_anchored = node_anchored
        self.generator = combined_syn.get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            dataset = combined_syn.get_dataset("graph", size // 2,
                np.arange(self.min_size + 1, self.max_size + 1))
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()) if \
                    use_distributed_sampling else None
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=sampler, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query,
        train):
        def sample_subgraph(graph, offset=0, use_precomp_sizes=False,
            filter_negs=False, supersample_small_graphs=False, neg_target=None,
            hard_neg_idxs=None):
            if neg_target is not None: graph_idx = graph.G.graph["idx"]
            use_hard_neg = (hard_neg_idxs is not None and graph.G.graph["idx"]
                in hard_neg_idxs)
            done = False
            n_tries = 0
            while not done:
                if use_precomp_sizes:
                    size = graph.G.graph["subgraph_size"]
                else:
                    if train and supersample_small_graphs:
                        sizes = np.arange(self.min_size + offset,
                            len(graph.G) + offset)
                        ps = (sizes - self.min_size + 2) ** (-1.1)
                        ps /= ps.sum()
                        size = stats.rv_discrete(values=(sizes, ps)).rvs()
                    else:
                        d = 1 if train else 0
                        size = random.randint(self.min_size + offset - d,
                            len(graph.G) - 1 + offset)
                start_node = random.choice(list(graph.G.nodes))
                neigh = [start_node]
                frontier = list(set(graph.G.neighbors(start_node)) - set(neigh))
                visited = set([start_node])
                # print(size)
                while len(neigh) < size:
                    new_node = random.choice(list(frontier))
                    assert new_node not in neigh
                    neigh.append(new_node)
                    visited.add(new_node)
                    frontier += list(graph.G.neighbors(new_node))
                    frontier = [x for x in frontier if x not in visited]
                if self.node_anchored:
                    anchor = neigh[0]
                    for v in graph.G.nodes:
                        graph.G.nodes[v]["node_feature"] = (torch.ones(1) if
                            anchor == v else torch.zeros(1))
                        #print(v, graph.G.nodes[v]["node_feature"])
                neigh = graph.G.subgraph(neigh)
                if use_hard_neg and train:
                    neigh = neigh.copy()
                    if random.random() < 1.0 or not self.node_anchored: # add edges
                        non_edges = list(nx.non_edges(neigh))
                        if len(non_edges) > 0:
                            for u, v in random.sample(non_edges, random.randint(1,
                                min(len(non_edges), 5))):
                                neigh.add_edge(u, v)
                    else:                         # perturb anchor
                        anchor = random.choice(list(neigh.nodes))
                        for v in neigh.nodes:
                            neigh.nodes[v]["node_feature"] = (torch.ones(1) if
                                anchor == v else torch.zeros(1))

                if (filter_negs and train and len(neigh) <= 6 and neg_target is
                    not None):
                    matcher = nx.algorithms.isomorphism.GraphMatcher(
                        neg_target[graph_idx], neigh)
                    if not matcher.subgraph_is_isomorphic(): done = True
                else:
                    done = True

            return graph, DSGraph(neigh)

        augmenter = feature_preprocess.FeatureAugment()

        pos_target = batch_target
        pos_target, pos_query = pos_target.apply_transform_multi(sample_subgraph)
        neg_target = batch_neg_target
        # TODO: use hard negs
        hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
            int(len(neg_target.G) * 1/2)))
        #hard_neg_idxs = set()
        batch_neg_query = Batch.from_data_list(
            [DSGraph(self.generator.generate(size=len(g))
                if i not in hard_neg_idxs else g)
                for i, g in enumerate(neg_target.G)])
        for i, g in enumerate(batch_neg_query.G):
            g.graph["idx"] = i
        _, neg_query = batch_neg_query.apply_transform_multi(sample_subgraph,
            hard_neg_idxs=hard_neg_idxs)
        if self.node_anchored:
            def add_anchor(g, anchors=None):
                if anchors is not None:
                    anchor = anchors[g.G.graph["idx"]]
                else:
                    anchor = random.choice(list(g.G.nodes))
                for v in g.G.nodes:
                    if "node_feature" not in g.G.nodes[v]:
                        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                            else torch.zeros(1))
                return g
            neg_target = neg_target.apply_transform(add_anchor)
        pos_target = augmenter.augment(pos_target).to(utils.get_device())
        pos_query = augmenter.augment(pos_query).to(utils.get_device())
        neg_target = augmenter.augment(neg_target).to(utils.get_device())
        neg_query = augmenter.augment(neg_query).to(utils.get_device())
        #print(len(pos_target.G[0]), len(pos_query.G[0]))
        return pos_target, pos_query, neg_target, neg_query

class OTFSynImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly synthetic data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}".format(str(self.node_anchored),
            self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = utils.batch_nx_graphs(pos_a)
            pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b

#### the DataSource we actually use
class DiskDataSource(DataSource):
    """ Uses a set of graphs saved in a dataset file to train the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.

    See the load_dataset function for supported datasets.
    """
    def __init__(self, dataset_name, node_anchored=False, min_size=5,
        max_size=29):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=6, min_size=2, seed=None,filter_negs=False, sample_method="tree-pair",N = 0):
        #the origin setting is max_size=15, min_size=5, current setting used for better visualization,not for training

        batch_size = a

        train_set, test_set, task = self.dataset
        graphs = train_set if train else test_set

        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []

        N =N % 32

        for i in range(batch_size // 2):
            pos_a.append(graphs[N * batch_size * 2 + 4 * i])
            pos_b.append(graphs[N * batch_size * 2 + 4 * i +1])
            neg_a.append(graphs[N * batch_size * 2 + 4 * i +2])
            neg_b.append(graphs[N * batch_size * 2 + 4 * i +3])

        pos_a = utils.batch_nx_graphs(pos_a, anchors=pos_a_anchors if
            self.node_anchored else None)
        pos_b = utils.batch_nx_graphs(pos_b, anchors=pos_b_anchors if
            self.node_anchored else None)
        neg_a = utils.batch_nx_graphs(neg_a, anchors=neg_a_anchors if
            self.node_anchored else None)
        neg_b = utils.batch_nx_graphs(neg_b, anchors=neg_b_anchors if
            self.node_anchored else None)
        return pos_a, pos_b, neg_a, neg_b

class DiskImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly real data.
    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, dataset_name, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0
        self.dataset = load_dataset(dataset_name)
        self.train_set, self.test_set, _ = self.dataset
        self.dataset_name = dataset_name

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            neighs = []
            for j in range(size // 2):
                graph, neigh = utils.sample_neigh(self.train_set if train else
                    self.test_set, random.randint(self.min_size, self.max_size))
                neighs.append(graph.subgraph(neigh))
            dataset = GraphDataset(neighs)
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=None, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}-{}".format(self.dataset_name.lower(),
            str(self.node_anchored), self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = utils.batch_nx_graphs(pos_a)
            pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    for name in ["enzymes", "reddit-binary", "cox2"]:
        data_source = DiskDataSource(name)
        train, test, _ = data_source.dataset
        i = 11
        neighs = [utils.sample_neigh(train, i) for j in range(10000)]
        clustering = [nx.average_clustering(graph.subgraph(nodes)) for graph,
            nodes in neighs]
        path_length = [nx.average_shortest_path_length(graph.subgraph(nodes))
            for graph, nodes in neighs]
        #plt.subplot(1, 2, i-9)
        plt.scatter(clustering, path_length, s=10, label=name)
    plt.legend()
    plt.savefig("plots/clustering-vs-path-length.png")
