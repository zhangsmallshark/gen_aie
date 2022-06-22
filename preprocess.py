import os
import numpy as np
from collections import OrderedDict
import math
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

import torch
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid, Reddit, Reddit2, PPI, Flickr, Yelp, AmazonProducts
from torch_geometric.transforms import NormalizeFeatures

from estimate_time import estimate_layer_time
# /home/fpga/Downloads/metis-5.1.0/build/Linux-x86_64/programs/gpmetis


def gen_metis_input(num_nodes, graph, base_dir, dataset):
    # coo
    edge_index = graph.edge_index
    edge_index = to_undirected(edge_index)
    edge_index = edge_index.tolist()
    num_edges = len(edge_index[0])
    adj = {}

    for i in range(num_edges):
        r = edge_index[0][i]
        c = edge_index[1][i]

        if r not in adj:
            adj[r] = [c]
        else:
            adj[r].append(c)

    connected_nodes = len(adj)

    added_edges = 0
    # for metis
    if connected_nodes != num_nodes:
        print(f'unconnected nodes {num_nodes-connected_nodes}')
        unconnected_nodes_file = f'{base_dir}/{dataset}/{dataset}_unconnected.txt'
        f1 = open(unconnected_nodes_file, 'w')
        for i in range(num_nodes):
            if i not in adj:
                adj[i] = [i]
                added_edges += 1
                f1.write(f'{i}\n')
        f1.close()

    file_path = f'{base_dir}/{dataset}/{dataset}.graph'
    f0 = open(file_path, 'w')
    f0.write(f'{num_nodes} {int(num_edges/2)+int(added_edges/2)}\n')
    for i in range(num_nodes):
        neigs = adj[i]
        for n in neigs:
            f0.write(f'{n+1} ')
        f0.write('\n')
    f0.close()

    adj_path = f'{base_dir}/{dataset}/{dataset}_adj.pkl'
    with open(adj_path, 'wb') as f2:
        pickle.dump(adj, f2, pickle.HIGHEST_PROTOCOL)


def gpmetis_clustering(num_nodes, num_parts, base_dir, dataset):
    partition_file = f'{base_dir}/{dataset}/{dataset}.graph.part.{num_parts}'
    par_dic = {}
    traversal_dic = {}
    root_l = []

    f0 = open(partition_file, 'r')
    for i in range(num_nodes):
        line = f0.readline()
        line = line.strip()
        if not line:
            print('Read error !!!')

        par_idx = int(line)
        if par_idx in par_dic:
            par_dic[par_idx].append(i)
        else:
            par_dic[par_idx] = [i]

    re_idx = 0
    for k in par_dic.keys():
        for v in par_dic[k]:
            traversal_dic[v] = re_idx
            re_idx += 1

        root_l.append([par_dic[k][0], len(par_dic[k])])

    f0.close()
    return traversal_dic, root_l


def reordering(traversal_order, root_l, base_dir, dataset):
    adj_path = f'{base_dir}/{dataset}/{dataset}_adj.pkl'
    with open(adj_path, 'rb') as f:
        adj = pickle.load(f)

    re_graph = {}
    for v in traversal_order.keys():
        idx = traversal_order[v]
        if idx not in re_graph:
            re_graph[idx] = []

        neighbours = adj[v]
        for neighbour in neighbours:
            n_idx = traversal_order[neighbour]
            re_graph[idx].append(n_idx)

    re_root_l = []
    for r in root_l:
        r_root = traversal_order[r[0]]
        re_root_l.append([r_root, r[1]])

    return re_graph, re_root_l


def show_adj(graph, re_root_l, partition):
    num_nodes = int(len(graph) / partition)
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    for v in range(num_nodes):
        neighbours = graph[v]
        for neighbour in neighbours:
            if neighbour >= num_nodes:
                continue
            adj_mat[v][neighbour] = 1

    # nnz = np.count_nonzero(adj_mat)
    # print(f'graph density is {nnz / (num_nodes * num_nodes)}')

    cmap = ListedColormap(['w', 'k'])
    # fig = plt.figure(figsize=(num_nodes, num_nodes), dpi=20)
    # plt.imshow(adj_mat, interpolation='nearest', cmap=cmap)
    plt.matshow(adj_mat, cmap=cmap)

    if len(re_root_l) > 0:
        ax = plt.gca()
        if len(re_root_l) > 100:
            num_rect = 100
        else:
            num_rect = len(re_root_l)

        for i in range(0, num_rect, 1):
            rect = patches.Rectangle((re_root_l[i][0], re_root_l[i][0]), re_root_l[i][1], re_root_l[i][1], 
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.colorbar()
    plt.show()


def calc_tile_density(mat, tile_size, dir):
    r_a, c_a = mat.shape
    row_tiles = int(r_a / tile_size)
    col_tiles = int(c_a / tile_size)

    f0 = open(f'{dir}/tile_density.txt', 'w')
    for i in range(row_tiles):
        s_r = tile_size * i
        e_r = tile_size * (i + 1)
        d_str = ''

        for j in range(col_tiles):
            s_c = tile_size * j
            e_c = tile_size * (j + 1)
            p_mat = mat[s_r:e_r, s_c:e_c]
            nnz = np.count_nonzero(p_mat)

            d_str += f'{nnz/(tile_size*tile_size):0.6f} '
        f0.write(d_str)
        f0.write('\n')
    f0.close


def save_partitions(graph, re_root_l, tile_size, base_dir, dataset):
    p_mat_dir = f'{base_dir}/{dataset}/p_mat'
    os.system(f'rm -rf {p_mat_dir}/*')
    f0 = open(f'{p_mat_dir}/p_density.txt', 'w')

    len_r = len(re_root_l)
    for i in range(len_r):
        s_v = re_root_l[i][0]
        len0 = re_root_l[i][1]
        len1 = math.ceil(len0 / tile_size) * tile_size
        e_v0 = s_v + len0
        e_v1 = s_v + len1
        p_adj = np.zeros((len1, len1), dtype=np.int8)
        nnz = 0

        for v in range(s_v, e_v0):
            neighbours = graph[v]
            for neighbour in neighbours:
                if neighbour >= e_v1 or neighbour < s_v:
                    continue
                p_adj[v-s_v][neighbour-s_v] = 1
                nnz += 1

        print(f'part {i}, length {len0}/{len1}, density {nnz/(len0*len0)}/{nnz/(len1*len1)}')
        f0.write(f'part {i}, length {len0}/{len1}, density {nnz/(len0*len0)}/{nnz/(len1*len1)}\n')
        p_mat_path = f'{p_mat_dir}/{i}.npy'
        np.save(p_mat_path, p_adj)

        calc_tile_density(p_adj, tile_size, p_mat_dir)
    
    f0.close()


if __name__ == "__main__":

    # g_name = 'Cora'

    # g_name = 'CiteSeer'

    # g_name = 'PubMed'

    # g_name = 'PPI'

    g_name = 'Flickr'

    # g_name = 'Yelp'

    # g_name = 'Reddit2'

    # g_name = 'Amazon'

    dataset_root = f'./datasets/{g_name}'

    if g_name == 'Cora':
        dataset = Planetoid(root='./datasets', name=g_name, transform=NormalizeFeatures())

    elif g_name == 'CiteSeer':
        dataset = Planetoid(root='./datasets', name=g_name, transform=NormalizeFeatures())

    elif g_name == 'PubMed':
        dataset = Planetoid(root='./datasets', name=g_name, transform=NormalizeFeatures())
        num_parts = 7

    elif g_name == 'PPI':
        dataset = PPI(root=dataset_root, split='train', transform=None)

    elif g_name == 'Flickr':
        dataset = Flickr(root=dataset_root, transform=None)
        num_parts = 28

    elif g_name == 'Yelp':
        dataset = Yelp(root=dataset_root, transform=None)
        num_parts = 225

    elif g_name == 'Reddit2':
        dataset = Reddit2(root=dataset_root)
        num_parts = 73

    elif g_name == 'Amazon':
        dataset = AmazonProducts(root=dataset_root)
        num_parts = 491

    print(f'\nDataset: {dataset}:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    num_classes = dataset.num_classes

    # Get the first graph object.
    g0 = dataset[0]
    # print(g0)

    # Gather some statistics about the graph.
    num_nodes = g0.num_nodes
    print(f'Number of nodes: {num_nodes}')
    print(f'Number of edges: {g0.num_edges}')

    # num_parts = math.ceil(num_nodes / 3200)
    # print(f'Number of partitions: {num_parts}')

    base_dir = './graph_mat'
    gen_metis_input(num_nodes, g0, base_dir, g_name)

    # traversal_order, root_l = gpmetis_clustering(num_nodes, num_parts, base_dir, g_name)
    # re_graph, re_root_l = reordering(traversal_order, root_l, base_dir, g_name)

    # h1 = 128
    # h2 = num_classes
    # r_a = num_nodes
    # c_a = num_nodes
    # c_x = dataset.num_features
    # c_w = h1
    # tile_size_a = 64
    # tile_size_b = 32
    # tile_size_x = 32
    # tile_size_w = 32
    # layer_time1 = estimate_layer_time(r_a, c_a, c_x, c_w, re_root_l, re_graph, tile_size_a, tile_size_b, tile_size_x, tile_size_w)

    # print('\n')
    # c_x = h1
    # c_w = h2
    # layer_time2 = estimate_layer_time(r_a, c_a, c_x, c_w, re_root_l, re_graph, tile_size_a, tile_size_b, tile_size_x, tile_size_w)
    # model_time = layer_time1 + layer_time2

    # print(f'layer_time1 : {layer_time1} us, layer_time2 {layer_time2} us')
    # print(f'model time: {model_time} us, {model_time / 1000} ms')

    # save_partitions(re_graph, re_root_l, tile_size, base_dir, g_name)
    # show_adj(re_graph, re_root_l, 1)
    print('preprocess finish !!!')
