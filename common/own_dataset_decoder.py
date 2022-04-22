import os
import os.path as osp
import pdb
import shutil
import glob

import torch.nn.functional as F
import numpy as np
import torch
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    # try:
    #     src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    #     src = torch.tensor(src, dtype=dtype).squeeze()
    # except:
    #     src = [[str(x) for x in line.split(sep)[start:end]] for line in src]
    #     # src = torch.tensor(src, dtype=dtype).squeeze()
    return src

def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)

def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')

        # pdb.set_trace()
    if 'node_labels' in names:
        # pdb.set_trace()
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        node_attr_graphIdx = read_file(folder, prefix, 'graph_indicator')
        node_labels = cat([node_labels.view(-1,1), node_attr_graphIdx.view(-1,1)])

        # if node_labels.dim() == 1:
        #     node_labels = node_labels.unsqueeze(-1)
        # # node_labels = node_labels - node_labels.min(dim=0)[0] #########
        # node_labels = node_labels.unbind(dim=-1)
        # #node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        # node_labels = torch.cat(node_labels, dim=-1).to(torch.int64)
        # pdb.set_trace()
    x = cat([node_attributes,node_labels]) ###@@ #fatgoose
    #node_shapeName = node_attributes

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.float)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        # edge_labels = edge_labels - edge_labels.min(dim=0)[0] ###################
        # edge_labels = edge_labels.unbind(dim=-1)
        #edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        # edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
        edge_labels = edge_labels.to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])

    #pdb.set_trace()

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    # fatgoose special for decoder: 0420
    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    # pdb.set_trace()

    return data, slices

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    # if data.node_shapeName is not None:
    #     slices['node_shapeName'] = data['node_shapeName']

    return data, slices



class OwnDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False):
        self.name = name
        self.cleaned = cleaned
        super(OwnDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # pdb.set_trace()
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))