import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
import tqdm

import torch.nn.functional as F
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)


class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            self.layers.append(dglnn.SAGEConv(in_dim, out_dim, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feature = g.ndata["features"]
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g,
                                th.arange(g.num_nodes()).cuda(),
                                sampler,
                                device="cuda",
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0,
                                use_uva=True)

        for l, layer in enumerate(self.layers):
            y = th.empty(g.num_nodes(),
                         self.n_hidden if l != len(self.layers) -
                         1 else self.n_classes,
                         dtype=feature.dtype)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feature[input_nodes.cpu()].cuda()
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to("cpu")
            feature = y
        return y


class GAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.relu,
                 feat_dropout=0.6,
                 attn_dropout=0.6):
        assert len(n_heads) == n_layers
        assert n_heads[-1] == 1

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads

        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden * n_heads[i - 1]
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            layer_activation = None if i == n_layers - 1 else activation
            self.layers.append(
                dglnn.GATConv(in_dim,
                              out_dim,
                              n_heads[i],
                              feat_drop=feat_dropout,
                              attn_drop=attn_dropout,
                              activation=layer_activation,
                              allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h

    def inference(self, g, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feature = g.ndata["features"]
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g,
                                th.arange(g.num_nodes()).cuda(),
                                sampler,
                                device="cuda",
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0,
                                use_uva=True)

        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = th.empty(g.num_nodes(),
                             self.n_classes * self.n_heads[l],
                             dtype=th.float32)
            else:
                y = th.empty(g.num_nodes(),
                             self.n_hidden * self.n_heads[l],
                             dtype=th.float32)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feature[input_nodes.cpu()].cuda()
                h = layer(blocks[0], x)
                if l == self.n_layers - 1:
                    h = h.mean(1)
                else:
                    h = h.flatten(1)
                y[output_nodes] = h.to("cpu")
            feature = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
