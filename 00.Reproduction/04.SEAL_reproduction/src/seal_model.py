"""CPU port of pinned muhanzhang/pytorch_DGCNN; see vendor/manifest.json.

The original row-normalized graph convolution is preserved. It is not PyG's
symmetric GCNConv. Only the obsolete compiled sparse-matrix extension is removed.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def initialize_linears(module):
    # Original weights_init changes Linear layers; Conv1d keeps torch defaults.
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)


def drnl(adjacency):
    """Original DRNL formula, computing only the two needed distance vectors."""
    from scipy.sparse.csgraph import shortest_path

    size = adjacency.shape[0]
    if size < 2:
        raise ValueError("An enclosing subgraph must contain both target nodes")
    without_source = adjacency[1:, 1:]
    keep = [0] + list(range(2, size))
    without_target = adjacency[keep, :][:, keep]
    d1 = shortest_path(without_source, directed=False, unweighted=True, indices=0)[1:]
    d0 = shortest_path(without_target, directed=False, unweighted=True, indices=0)[1:]
    # Preserve the released integer casts/sentinel filters, including overflow behavior.
    with np.errstate(invalid="ignore", over="ignore"):
        half, remainder = np.divmod((d0 + d1).astype(np.int64), 2)
        labels = 1 + np.minimum(d0, d1).astype(np.int64) + half * (half + remainder - 1)
    labels = np.concatenate((np.asarray([1, 1], dtype=np.int64), labels))
    labels[(labels > 1_000_000) | (labels < -1_000_000)] = 0
    return labels


def enclosing_subgraph(adjacency, source, target, hops=1):
    """Keep every h-hop neighbor. Target endpoints occupy local positions 0,1."""
    if source == target:
        raise ValueError("Target endpoints must differ")
    visited = {int(source), int(target)}
    fringe = set(visited)
    for _ in range(hops):
        neighbors = set()
        for node in sorted(fringe):
            neighbors.update(adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]])
        fringe = neighbors - visited
        visited.update(fringe)
        if not fringe:
            break
    nodes = np.asarray([source, target] + sorted(visited - {source, target}), dtype=np.int64)
    subgraph = adjacency[nodes, :][:, nodes].tocsr()
    # In this experiment all DDA labels are outside the graph; also enforce this locally.
    if subgraph[0, 1] or subgraph[1, 0]:
        subgraph = subgraph.tolil()
        subgraph[0, 1] = subgraph[1, 0] = 0
        subgraph = subgraph.tocsr()
        subgraph.eliminate_zeros()
    labels = drnl(subgraph)
    row, col = subgraph.nonzero()
    edges = np.asarray([row, col], dtype=np.int64)
    return nodes, edges, labels


def collate_graphs(graphs, node_features, label_count):
    features, edges, targets, sizes = [], [], [], []
    offset = 0
    for nodes, edge_index, labels, target in graphs:
        structural = F.one_hot(torch.as_tensor(labels, dtype=torch.long), label_count).float()
        features.append(torch.cat([structural, node_features[nodes]], dim=1))
        edges.append(torch.as_tensor(edge_index, dtype=torch.long) + offset)
        targets.append(target)
        sizes.append(len(nodes))
        offset += len(nodes)
    return (torch.cat(features), torch.cat(edges, dim=1), sizes,
            torch.tensor(targets, dtype=torch.long))


class DGCNN(nn.Module):
    """Original [32,32,32,1] DGCNN + two-class MLP, without native extensions."""

    def __init__(self, input_channels, k, latent=(32, 32, 32, 1), hidden=128):
        super().__init__()
        if k < 10:
            raise ValueError("Original Conv1d head requires SortPooling k >= 10")
        self.k = int(k)
        widths = [input_channels] + list(latent)
        self.convs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        total = sum(latent)
        self.conv1 = nn.Conv1d(1, 16, total, total)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 5, 1)
        dense = (int((self.k - 2) / 2 + 1) - 5 + 1) * 32
        initialize_linears(self)
        self.hidden = nn.Linear(dense, hidden)
        self.output = nn.Linear(hidden, 2)
        initialize_linears(self.hidden)
        initialize_linears(self.output)

    def forward(self, features, edge_index, sizes):
        src, dst = edge_index
        degree = torch.bincount(dst, minlength=len(features)).to(features.dtype).unsqueeze(1) + 1
        states = []
        for layer in self.convs:
            aggregated = features.clone()
            aggregated.index_add_(0, dst, features[src])
            features = torch.tanh(layer(aggregated) / degree)
            states.append(features)
        states = torch.cat(states, dim=1)
        pooled = []
        for graph in torch.split(states, sizes):
            indices = graph[:, -1].topk(min(self.k, len(graph))).indices
            selected = graph.index_select(0, indices)
            pooled.append(F.pad(selected, (0, 0, 0, self.k - len(selected))))
        pooled = torch.stack(pooled).reshape(len(sizes), 1, -1)
        convolved = F.relu(self.conv2(self.pool(F.relu(self.conv1(pooled)))))
        hidden = F.relu(self.hidden(convolved.flatten(1)))
        return F.log_softmax(self.output(F.dropout(hidden, p=0.5, training=self.training)), dim=1)


def pooling_k(sizes, percentile=0.6):
    values = sorted(map(int, sizes))
    if not values or not 0 < percentile <= 1:
        raise ValueError("SortPooling requires nonempty graph sizes and a valid percentile")
    return max(10, values[math.ceil(percentile * len(values)) - 1])
