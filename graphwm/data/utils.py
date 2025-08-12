# some utils are from VGPL/DPI.
import os
import json
import numpy as np
import torch
import h5py

from torch_geometric.nn import MessagePassing


class ConnectedComponents(MessagePassing):
    def __init__(self):
        super().__init__(aggr="max")

    def forward(self, n_node, edge_index):
        x = torch.arange(n_node, device=edge_index.device).view(-1, 1)
        last_x = torch.zeros_like(x)

        while not x.equal(last_x):
            last_x = x.clone()
            x = self.propagate(edge_index, x=x)
            x = torch.max(x, last_x)

        unique, perm = torch.unique(x, return_inverse=True)
        perm = perm.view(-1)
        return unique, perm

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


def store_data(data_names, data, path):
    """Save arrays to h5 file."""
    with h5py.File(path, 'w') as hf:
        for name, d in zip(data_names, data):
            hf.create_dataset(name, data=d)


def load_data(data_names, path):
    """Load arrays from h5 file."""
    with h5py.File(path, 'r') as hf:
        data = [np.array(hf.get(name)) for name in data_names]
    return data


def load_data_w_idx(data_names, path, select_idx):
    """Load slices of arrays from h5 file according to select_idx."""
    with h5py.File(path, 'r') as hf:
        data = [np.array(hf.get(name)[select_idx]) for name in data_names]
    return data


def read_metadata(data_path):
    """Read json metadata file."""
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.load(fp)


def dict_collate_fn(batch):
    """
    Collate function to combine a list of dicts into one dict for DataLoader.
    Takes care of incrementing bond, cluster, and component indices offsets for batching.
    """
    batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}

    # Adjust bonds indices by node offsets
    if 'bonds' in batch:
        index_offsets = torch.cumsum(batch['n_bond'], dim=0)
        node_offsets = torch.cumsum(batch['n_particle'], dim=0)
        for i in range(len(index_offsets) - 1):
            batch['bonds'][index_offsets[i]:index_offsets[i + 1]] += node_offsets[i]

    # Adjust cluster/grouping offsets
    if 'n_keypoint' in batch:
        if 'n_component' in batch:
            component_offsets = torch.cumsum(batch['n_component'], dim=0)
        index_offsets = torch.cumsum(batch['n_keypoint'], dim=0)
        node_offsets = torch.cumsum(batch['n_particle'], dim=0)
        bond_offset = torch.cumsum(batch['n_cg_bond'], dim=0)
        for i in range(len(index_offsets) - 1):
            if 'n_component' in batch:
                batch['component'][node_offsets[i]:node_offsets[i + 1]] += component_offsets[i]
            if 'keypoint' in batch:
                batch['keypoint'][index_offsets[i]:index_offsets[i + 1]] += node_offsets[i]
            batch['cluster'][node_offsets[i]:node_offsets[i + 1]] += index_offsets[i]
            batch['cg_bonds'][bond_offset[i]:bond_offset[i + 1]] += index_offsets[i]

    return batch
