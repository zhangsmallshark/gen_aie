from collections import OrderedDict, defaultdict
import pickle as pkl
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import math
import networkx as nx
import nxmetis
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
# dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# from torch_geometric.datasets import KarateClub
# dataset = KarateClub()

# from torch_geometric.datasets import Reddit
# d_path = './data/Reddit'
# dataset = Reddit(d_path)

# from torch_geometric.datasets import Flickr
# d_path = './data/Flickr'
# dataset = Flickr(d_path)


def reordering0(graph_x, traversal_order, root_l):
    adj = graph_x.adj
    n_graph = defaultdict(list)
    num_nodes = len(adj)
    n_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # for v in traversal_order.keys():
    #     r_v = traversal_order[v]

    for v, r_v in enumerate(traversal_order):
        if r_v not in n_graph:
            n_graph[r_v] = []

        neighbours = adj[v]
        for neighbour in neighbours.keys():
            n_idx = traversal_order[neighbour]
            n_graph[r_v].append(n_idx)
            n_adj[r_v][n_idx] = 1.0

    re_root_l = []
    for r in root_l:
        r_root = traversal_order[r[0]]
        re_root_l.append([r_root, r[1]])

    return n_graph, n_adj, re_root_l


def enlarge_nonzeros(adj_mat):
    rows, cols = adj_mat.shape
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    for i in range(rows):
        for j in range(cols):
            if adj_mat[i, j] == 1.0:
                for d in directions:
                    n_i = i + d[0]
                    n_j = j + d[1]
                    if 0 <= n_i < rows and 0 <= n_j < cols:
                        adj_mat[n_i, n_j] = 1.1


def show_adj0(graph):
    edge_index = graph.edge_index
    num_edges = graph.num_edges
    num_nodes = graph.num_nodes
    adj_mat = np.zeros((num_nodes, num_nodes))

    for idx in range(num_edges):
        v0 = edge_index[0][idx]
        v1 = edge_index[1][idx]
        adj_mat[v0][v1] = 1

    cmap = ListedColormap(['w', 'r'])
    plt.matshow(adj_mat, cmap=cmap)
    # plt.matshow(adj_mat, cmap=plt.cm.gray_r)

    plt.colorbar()
    plt.show()


def show_adj1(graph, re_root_l):
    num_nodes = len(graph)
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for v in graph.keys():
        neighbours = graph[v]
        for neighbour in neighbours:
            adj_mat[v][neighbour] = 1.0

    enlarge_nonzeros(adj_mat)
    cmap = ListedColormap(['w', '#2d1c4d'])
    plt.figure(figsize=(16, 16), dpi=128)
    # plt.imshow(adj_mat, interpolation='nearest', cmap=cmap)
    plt.matshow(adj_mat, cmap=cmap)

    if len(re_root_l) > 0:
        ax = plt.gca()
        if len(re_root_l) > 100:
            num_rect = 100
        else:
            num_rect = len(re_root_l)

        for i in range(0, num_rect, 1):
            rect = patches.Rectangle((re_root_l[i][0], re_root_l[i][0]), re_root_l[i][1], re_root_l[i][1], linewidth=0.4,
                                     edgecolor='#e7ddb8', facecolor='none')
            ax.add_patch(rect)
    # plt.colorbar()
    plt.show()
    # plt.savefig('/home/fpga/Desktop/r3.png', bbox_inches='tight', pad_inches=0)
    # plt.savefig('/home/fpga/Desktop/r5.png')

    return adj_mat


def show_adj2(graph, re_root_l, partition):
    num_nodes = int(len(graph) / partition)
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    # for v in graph.keys():
    for v in range(num_nodes):
        neighbours = graph[v]
        for neighbour in neighbours:
            if neighbour >= num_nodes:
                continue
            adj_mat[v][neighbour] = 1

    cmap = ListedColormap(['w', 'k'])
    # plt.figure(figsize=(num_nodes, num_nodes), dpi=20)
    # plt.imshow(adj_mat, interpolation='nearest', cmap=cmap)
    plt.matshow(adj_mat, cmap=cmap)

    if len(re_root_l) > 0:
        ax = plt.gca()
        if len(re_root_l) > 100:
            num_rect = 100
        else:
            num_rect = len(re_root_l)

        for i in range(0, num_rect, 1):
            rect = patches.Rectangle((re_root_l[i][0], re_root_l[i][0]), re_root_l[i][1], re_root_l[i][1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.colorbar()
    plt.show()
    return adj_mat


def show_adj3(adj, idx):
    # enlarge_nonzeros(adj)
    cmap = ListedColormap(['w', 'r'])
    plt.matshow(adj, cmap=cmap)
    # plt.matshow(adj_mat, cmap=plt.cm.gray_r)

    plt.colorbar()
    plt.show()
    # plt.savefig(f'pic{idx}.png')


def louvain_clustering(graph_x):
    """
    Clustering the graph with louvain.
    """
    # compute the best partition
    partition = community_louvain.best_partition(graph_x)

    degrees = graph_x.degree()
    cluster_dic = {}
    for v, p in partition.items():
        if p in cluster_dic:
            cluster_dic[p].append([v, degrees[v]])
        else:
            cluster_dic[p] = [[v, degrees[v]]]

    num_nodes = len(graph_x.nodes)
    traversal_order = np.zeros(num_nodes, dtype=np.int32)
    traversal_dic = OrderedDict()

    root_l = []
    idx = 0
    for p, v_l in cluster_dic.items():
        # v_l.sort(key=lambda x: x[1], reverse=True)
        for v in v_l:
            traversal_order[v[0]] = idx
            traversal_dic[v[0]] = idx
            idx += 1

        len_v = len(v_l)
        root_l.append([v_l[0][0], len_v])

    return traversal_order, traversal_dic, root_l


def metis_partition(graph_x, cluster_number):
    """
    Partition the graph with Metis.
    """
    obj_val, parts = nxmetis.partition(graph_x, cluster_number)
    num_nodes = len(graph_x.nodes)
    traversal_order = np.zeros(num_nodes, dtype=np.int32)
    root_l = []
    color = []
    idx = 0
    for c_idx, c in enumerate(parts):
        for v in c:
            traversal_order[v] = idx
            idx += 1

        root_l.append([c[0], len(c)])
        color.extend([c_idx] * len(c))

    return traversal_order, root_l


def gen_metis_input(graph_x, num_nodes, num_edges, file_name):
    adj = graph_x.adj
    f0 = open(file_name, 'w')
    # f0.write(f'# Nodes: {num_nodes}	Edges: {int(num_edges/2)}\n')
    f0.write(f'{num_nodes} {int(num_edges / 2)}\n')
    for i in range(num_nodes):
        neigs = adj[i]
        neig_l = neigs.keys()
        for n in neig_l:
            f0.write(f'{n+1} ')
        f0.write('\n')

    f0.close()


def gpmetis_clustering(res_name, num_nodes):
    par_dic = {}
    traversal_order = []
    traversal_dic = OrderedDict()
    root_l = []

    f0 = open(res_name, 'r')
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

    for k in par_dic.keys():
        traversal_order.extend(par_dic[k])
        root_l.append([par_dic[k][0], len(par_dic[k])])

    for i, v in enumerate(traversal_order):
        traversal_dic[v] = i

    return traversal_dic, root_l


def degree_sort(graph_x):
    degree = graph_x.degree
    degree = [[ver, val] for (ver, val) in degree]
    num_ver = len(degree)
    degree.sort(key=lambda x: x[1], reverse=True)

    traversal_order = [0] * num_ver
    for idx, d in enumerate(degree):
        traversal_order[d[0]] = idx

    return traversal_order



if __name__ == "__main__":

    print(f'\nDataset: {dataset}:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Get the first graph object.
    g0 = dataset[0]
    x = g0.x.numpy()
    y = g0.y.numpy()
    train_mask = g0.train_mask.numpy()
    val_mask = g0.val_mask.numpy()
    test_mask = g0.test_mask.numpy()

    # np.save('feature_vector.npy', x)

    # Gather some statistics about the graph.
    # print(f'Number of nodes: {g0.num_nodes}')
    # print(f'Number of edges: {g0.num_edges}')
    # print(f'Number of training nodes: {g0.train_mask.sum()}')
    # print(f'Number of test nodes: {g0.test_mask.sum()}')

    # show_adj0(g0)

    graph_x = to_networkx(g0, to_undirected=True)

    # traversal_order, traversal_dic, root_l = louvain_clustering(graph_x)
    # traversal_order = degree_sort(graph_x)
    # root_l = [[0, 200]]
    # graph, n_adj, re_root_l = reordering0(graph_x, traversal_order, root_l)
    # adj_mat = show_adj1(graph, re_root_l)

    num_classes = 14
    traversal_order, root_l = metis_partition(graph_x, num_classes)
    graph, n_adj, re_root_l = reordering0(graph_x, traversal_order, root_l)
    adj_mat = show_adj1(graph, re_root_l)

    # gen_metis_input(graph_x, data.num_nodes, data.num_edges, './g_d0/Reddit.txt')
    # traversal_order, root_l = gpmetis_clustering('./g_d0/Reddit.txt.part.40', data.num_nodes)
    # graph, re_root_l = reordering(graph_x, traversal_order, root_l)
    # adj_mat = show_adj2(graph, re_root_l, 8)

    # gen_metis_input(graph_x, data.num_nodes, data.num_edges, './g_d0/cora.txt')
    # traversal_order, root_l = gpmetis_clustering('./g_d0/cora.txt.part.16', data.num_nodes)
    # graph, re_root_l = reordering(graph_x, traversal_order, root_l)
    # adj_mat = show_adj2(graph, re_root_l, 1)
