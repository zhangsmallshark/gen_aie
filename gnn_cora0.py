import math
import os.path
import networkx as nx
import nxmetis
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import numpy as np
import struct
from collections import OrderedDict
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# from torch_geometric.datasets import KarateClub
# dataset = KarateClub()

# from torch_geometric.datasets import Reddit
# d_path = './data/Reddit'
# dataset = Reddit(d_path)

tile_size0 = 8


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(model, data):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss


def test(model, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      print(f'Test Accuracy: {test_acc:.4f}')
      return test_acc


def decimal_converter(num):
    while num > 1:
        num /= 10
    return num


def float_to_bin(number, places=3):
    whole, dec = str(number).split(".")
    whole = int(whole)
    dec = int(dec)
    res = bin(whole).lstrip("0b") + "."
    for x in range(places):
        whole, dec = str((decimal_converter(dec)) * 2).split(".")
        dec = int(dec)
        res += whole

    return res


def reordering0(graph_x, traversal_order, root_l):
    adj = graph_x.adj
    graph0 = {}
    for idx, v in enumerate(traversal_order):
        if idx not in graph0:
            graph0[idx] = []

        neighbours = adj[v]
        for neighbour in neighbours.keys():
            n_idx = traversal_order.index(neighbour)
            graph0[idx].append(n_idx)

    re_root_l = []
    for r in root_l:
        r_root = traversal_order.index(r[0])
        re_root_l.append([r_root, r[1]])

    return graph0, re_root_l


def reordering1(graph_x, traversal_order, root_l):
    adj = graph_x.adj
    graph0 = {}
    for v in traversal_order.keys():
        idx = traversal_order[v]
        if idx not in graph0:
            graph0[idx] = []

        neighbours = adj[v]
        for neighbour in neighbours.keys():
            n_idx = traversal_order[neighbour]
            graph0[idx].append(n_idx)

    re_root_l = []
    for r in root_l:
        r_root = traversal_order[r[0]]
        re_root_l.append([r_root, r[1]])

    return graph0, re_root_l


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
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    for v in graph.keys():
        neighbours = graph[v]
        for neighbour in neighbours:
            adj_mat[v][neighbour] = 1

    cmap = ListedColormap(['w', '#2d1c4d'])
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
            rect = patches.Rectangle((re_root_l[i][0], re_root_l[i][0]), re_root_l[i][1], re_root_l[i][1], linewidth=1,
                                     edgecolor='#e7ddb8', facecolor='none')
            ax.add_patch(rect)
    plt.colorbar()
    plt.show()
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
            rect = patches.Rectangle((re_root_l[i][0], re_root_l[i][0]), re_root_l[i][1], re_root_l[i][1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.colorbar()
    plt.show()
    return adj_mat


def save_partitions(graph, re_root_l, base_name):
    len_r = len(re_root_l)

    for i in range(len_r):
        s_v = re_root_l[i][0]
        len0 = re_root_l[i][1]
        len1 = math.ceil(len0 / 32) * 32
        e_v0 = s_v + len0
        e_v1 = s_v + len1
        p_adj = np.zeros((len1, len1), dtype=np.int8)

        for v in range(s_v, e_v0):
            neighbours = graph[v]
            for neighbour in neighbours:
                if neighbour >= e_v1 or neighbour < s_v:
                    continue
                p_adj[v-s_v][neighbour-s_v] = 1

        print(f'part {i}, length {len0}')
        full_name = f'{base_name}{i}.npy'
        np.save(full_name, p_adj)


# def visualize0(h, color, epoch=None, loss=None):
#     plt.figure(figsize=(7, 7))
#     plt.xticks([])
#     plt.yticks([])
#
#     if torch.is_tensor(h):
#         h = h.detach().cpu().numpy()
#         plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
#         if epoch is not None and loss is not None:
#             plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
#     else:
#         nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False, node_size=40,
#                          node_color=color, cmap="Set2")
#     plt.show()


def cluster_pos(num_p, cycle_step, rad_base, rad_step):

    num_round = num_p % cycle_step + 1
    angs = np.linspace(0, 2 * np.pi, 1 + cycle_step)

    pos = []
    for i in range(num_round):
        rad_c = rad_base + rad_step * i
        for ang in angs:
            if ang > 0:
                pos.append(np.array([rad_c * np.cos(ang), rad_c * np.sin(ang)]))

    return pos


def visualize0(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        pos = nx.circular_layout(h)
        # pos = nx.spring_layout(h, seed=46)

        # angs = np.linspace(0, 2 * np.pi, max(color) + 2)
        # base = []
        # # radius of graph
        # rad_g = 6
        # for ang in angs:
        #     if ang > 0:
        #         base.append(np.array([rad_g * np.cos(ang), rad_g * np.sin(ang)]))
        #
        # for idx, c in enumerate(color):
        #     pos[idx] += base[c]

        # degrees = h.degree()
        # cluster_dic = {}
        # for idx, c in enumerate(color):
        #     if c in cluster_dic:
        #         cluster_dic[c].append([idx, degrees[idx]])
        #     else:
        #         cluster_dic[c] = [[idx, degrees[idx]]]
        #
        # for k, v in cluster_dic.items():
        #     v.sort(key=lambda x: x[1], reverse=True)
        #     len_v = len(v)
        #     c_pos = cluster_pos(len_v, 64, 0.4, 0.2)
        #
        #     vertex = v[0][0]
        #     pos[vertex][:] = base[k][:]
        #     for i in range(1, len_v):
        #         vertex = v[i][0]
        #         c_pos[i-1] += base[k]
        #         pos[vertex][:] = c_pos[i-1][:]

        # nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False, node_size=40,
        #                  node_color=color, cmap="Set2")

        nx.draw_networkx(h, pos=pos, with_labels=False, node_size=60,
                         node_color=color, cmap="Set2")

        # color the nodes according to their partition
        # cmap = cm.get_cmap('viridis', max(color) + 1)
        # nx.draw_networkx_nodes(h, pos, node_size=40, cmap=cmap, node_color=color)
        # nx.draw_networkx_edges(h, pos, alpha=0.5)

    plt.show()


def visualize1(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def metis_partition(graph, cluster_number):
    """
    Partition the graph with Metis.
    """
    obj_val, parts = nxmetis.partition(graph, cluster_number)

    traversal_order = []
    root_l = []
    color = []
    for c_idx, c in enumerate(parts):
        traversal_order.extend(c)
        root_l.append([c[0], len(c)])
        color.extend([c_idx] * len(c))

    # visualize0(graph, color=color)

    return traversal_order, root_l


def louvain_clustering(graph, label):
    """
    Clustering the graph with louvain.
    """
    # compute the best partition
    partition = community_louvain.best_partition(graph)

    # draw the graph
    # pos = nx.spring_layout(graph)
    pos = nx.circular_layout(graph)

    angs = np.linspace(0, 2 * np.pi, len(set(partition.values())) + 1)
    base = []
    # radius of graph
    rad_g = 6
    for ang in angs:
        if ang > 0:
            base.append(np.array([rad_g * np.cos(ang), rad_g * np.sin(ang)]))

    degrees = graph.degree()
    cluster_dic = {}
    for k, v in partition.items():
        if v in cluster_dic:
            cluster_dic[v].append([k, degrees[k]])
        else:
            cluster_dic[v] = [[k, degrees[k]]]

    traversal_order = []
    root_l = []
    for k, v in cluster_dic.items():
        # v.sort(key=lambda x: x[1], reverse=True)
        for ver in v:
            traversal_order.append(ver[0])

        len_v = len(v)
        root_l.append([v[0][0], len_v])
        c_pos = cluster_pos(len_v, 360, 1, 1)
        vertex = v[0][0]
        pos[vertex][:] = base[k][:]
        for i in range(1, len_v):
            vertex = v[i][0]
            c_pos[i-1] += base[k]
            pos[vertex][:] = c_pos[i-1][:]

    # color the nodes according to their partition
    # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    # nx.draw_networkx_nodes(graph, pos, partition.keys(), node_size=40,
    #                        cmap=cmap, node_color=list(partition.values()))

    # nx.draw_networkx_nodes(graph, pos, partition.keys(), node_size=40,
    #                        cmap="Set2", node_color=label)
    # nx.draw_networkx_edges(graph, pos, alpha=0.5)
    # plt.show()

    return traversal_order, root_l


def degree_sort(graph_x):
    degree = graph_x.degree
    degree = [[ver, val] for (ver, val) in degree]
    adj = graph_x.adj
    num_ver = len(degree)
    adj_mat = np.zeros((num_ver, num_ver))
    degree.sort(key=lambda x: x[1], reverse=True)

    traversal_order = []
    for d in degree:
        traversal_order.append(d[0])

    return traversal_order


def bdfs(graph_x, max_depth):
    adj = graph_x.adj
    num_nodes = len(adj)
    visited = []
    for root in range(num_nodes):
        if root not in visited:
            dfs(adj, root, visited, 0, max_depth)

    return visited


def dfs(adj, v, visited, cur_depth, max_depth):
    visited.append(v)
    if cur_depth > max_depth:
        return
    else:
        neighbours = adj[v]
        for neighbour in neighbours.keys():
            if neighbour not in visited:
                dfs(adj, neighbour, visited, cur_depth + 1, max_depth)


def bbfs(graph_x, max_depth):
    degree = graph_x.degree
    degree = [[ver, val] for (ver, val) in degree]
    degree.sort(key=lambda x: x[1], reverse=True)
    order = []
    for d in degree:
        order.append(d[0])

    adj = graph_x.adj
    num_nodes = len(adj)
    visited = []
    root_l = []

    # for root in range(num_nodes):
    for root in order:
        if root not in visited:
            cur_depth = bfs(adj, root, visited, 0, max_depth)
            root_l.append([root, cur_depth])

    return visited, root_l


def bfs(adj, v, visited, cur_depth, max_depth):
    queue = []
    visited.append(v)
    queue.append(v)

    while queue:
        if cur_depth > max_depth:
            break
        else:
            v = queue.pop(0)
            neighbours = adj[v]
            for neighbour in neighbours.keys():
                if neighbour not in visited:
                    cur_depth += 1
                    visited.append(neighbour)
                    queue.append(neighbour)

    return cur_depth


def get_neighbours(graph_x, v):
    n_l = []
    d_l = []
    degree = graph_x.degree
    adj = graph_x.adj
    neighbours = adj[v]
    for neighbour in neighbours.keys():
        d_l.append([neighbour, degree[neighbour]])

    d_l.sort(key=lambda x: x[1], reverse=True)
    for d in d_l:
        n_l.append(d[0])

    return n_l


def degree_bfs(graph_x, level_th, percent):
    degree = graph_x.degree
    degree = [[ver, val] for (ver, val) in degree]
    degree.sort(key=lambda x: x[1], reverse=True)
    order = []
    for d in degree:
        order.append(d[0])

    adj = graph_x.adj
    num_nodes = len(adj)
    count_nodes = 0
    level = 0
    visited = []
    queue = []
    # queue.append(order[0])

    while count_nodes < num_nodes:
        for root in order:
            if root not in visited:
                queue.append(root)
                break

        while queue:
            level_size = len(queue)
            while level_size > 0:
                level_size -= 1
                v = queue.pop(0)
                if v not in visited:
                    count_nodes += 1
                    visited.append(v)
                    n_l = get_neighbours(graph_x, v)
                    th0 = int(len(n_l) * percent)
                    queue.extend(n_l[th0:])

            if level == level_th:
                queue.clear()
                break

            level += 1

    return visited


def write_partial_mat_a(file_name, adj_mat, s_p, length):
    p_mat = adj_mat[s_p:s_p+length, s_p:s_p+length]
    m, r = divmod(length, 8)
    p_mat_8 = np.zeros(((m+1)*8, (m+1)*8))
    p_mat_8[0:length, 0:length] = p_mat[:, :]
    p_mat_8 = np.reshape(p_mat_8, (-1, 2))
    row, col = p_mat_8.shape

    f = open(file_name, "w")
    for i in range(row):
        f.write(f'{p_mat_8[i][0]}  {p_mat_8[i][1]}\n')
    f.close()


def write_partial_mat_b(file_name, b, s_p, length):
    p_mat = b[s_p:s_p+length, :]
    row0, col0 = p_mat.shape
    m, r = divmod(length, 8)
    p_mat_8 = np.zeros(((m+1)*8, col0))
    p_mat_8[0:length, :] = p_mat[:, :]
    p_mat_8 = np.reshape(p_mat_8, (-1, 2))
    row1, col1 = p_mat_8.shape

    f = open(file_name, "w")
    for i in range(row1):
        f.write(f'{p_mat_8[i][0]}  {p_mat_8[i][1]}\n')
    f.close()


# X = AXW
def xw(x_npy, w_npy, s_p, length):
    x = np.load(x_npy)
    w = np.load(w_npy)
    b = np.dot(x, w.transpose())
    file_name = 'p_mat_b.txt'
    write_partial_mat_b(file_name, b, s_p, length)


def p_mac(p_a, p_b, col_b):
    acc = [[0] * col_b for _ in range(8)]
    for i in range(8):
        for j in range(8):
            for k in range(col_b):
                acc[i][k] += p_a[0][i*8+j] * p_b[0][j*col_b+k]

    return np.array(acc)


def p_add(acc0, acc1):
    return np.add(acc0, acc1)


def read_partial_mat_a(adj_mat, s_p, length):
    p_mat = np.copy(adj_mat[s_p:s_p+length, s_p:s_p+length])
    zeros = np.zeros((length, length))
    adj_mat[s_p:s_p+length, s_p:s_p+length] = zeros[:, :]
    m, r = divmod(length, 8)
    p_mat_8 = np.zeros(((m+1)*8, (m+1)*8))
    p_mat_8[0:length, 0:length] = p_mat[:, :]
    # p_mat_8 = np.reshape(p_mat_8, (1, -1))
    return p_mat_8, m+1


def read_partial_mat_b(b, s_p, length):
    p_mat = b[s_p:s_p+length, :]
    row0, col0 = p_mat.shape
    m, r = divmod(length, 8)
    p_mat_8 = np.zeros(((m+1)*8, col0))
    p_mat_8[0:length, :] = p_mat[:, :]
    # p_mat_8 = np.reshape(p_mat_8, (1, -1))
    return p_mat_8


def compare_mat(a, b):
    dif = np.subtract(a, b)
    dif = np.absolute(dif)
    max_dif = np.amax(dif)
    if max_dif == 0:
        print('No dif !!')
    else:
        print(f'Max dif is {max_dif} !!')


def ab(adj_mat, b, re_root_l):
    adj_mat_copy = np.copy(adj_mat)
    b_copy = np.copy(b)
    refer_res = np.dot(adj_mat_copy, b_copy)
    num_root = len(re_root_l)
    row_a, col_a = adj_mat.shape
    row_b, col_b = b.shape
    res = np.zeros((row_a, col_b))

    for r in range(0, num_root, 1):
        s_p = re_root_l[r][0]
        length = re_root_l[r][1]
        p_a, m = read_partial_mat_a(adj_mat, s_p, length)
        p_b = read_partial_mat_b(b, s_p, length)
        p_res = np.zeros((m*8, col_b))

        for i in range(0, m*8, 8):
            p_acc1 = np.zeros((8, col_b))
            for j in range(0, m*8, 8):
                s_a = p_a[i:i+8, j:j+8]
                s_a = np.reshape(s_a, (1, -1))
                s_b = p_b[j:j+8, :]
                s_b = np.reshape(s_b, (1, -1))
                p_acc0 = p_mac(s_a, s_b, col_b)
                p_acc1 = p_add(p_acc0, p_acc1)

            p_res[i:i+8, :] = p_acc1[:, :]

        res[s_p:s_p+length, :] = p_res[0:length, :]

    rem_mat = np.dot(adj_mat, b)
    res = np.add(rem_mat, res)
    compare_mat(refer_res, res)


def gen_nse(graph_x, file_name):
    # ./bin/Release/pscan -f NSE -o /home/fpga/Desktop/graph_files/graph/g_d0/pscan_clustering_cora.txt /home/fpga/Desktop/graph_files/graph/g_d0/cora.nse
    edges = graph_x.edges
    f0 = open(file_name, 'w')
    f0.write(f'# Nodes: {len(graph_x.nodes)}	Edges: {len(edges)*2}\n')
    for e in edges:
        f0.write(f'{e[0]+1} {e[1]+1}\n')


def pscan_clustering(res_name, num_nodes):
    f0 = open(res_name, 'r')
    lines = f0.readlines()
    len_lines = len(lines)
    traversal_order = []
    root_l = []
    color = []
    for i in range(3, len_lines):
        # print(i)
        line = lines[i]
        line = line.strip()

        if len(line) > 0 and not line.startswith('#'):
            c0 = line.split(' ')
            c1 = [int(v)-1 for v in c0]
            traversal_order.extend(c1)
            root_l.append([c1[0], len(c1)])
            color.extend([i] * len(c1))

    unique_order = []
    for n in traversal_order:
        if n not in unique_order:
            unique_order.append(n)

    n_l0 = range(0, num_nodes)
    rem = []
    for n in n_l0:
        if n not in unique_order:
            rem.append(n)

    unique_order.extend(rem)

    # visualize0(graph, color=color)
    return unique_order, root_l


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

    # unique_order = []
    # for n in traversal_order:
    #     if n not in unique_order:
    #         unique_order.append(n)

    # n_l0 = range(0, num_nodes)
    # rem = []
    # for n in n_l0:
    #     if n not in unique_order:
    #         rem.append(n)

    # unique_order.extend(rem)

    # visualize0(graph, color=color)
    return traversal_dic, root_l


if __name__ == "__main__":
    # p_mat_a_f = 'p_mat_a.txt'
    # x_npy = './feature_vector.npy'
    # w_npy = './conv1_lin_weight.npy'
    # if i == 1:
    #     write_partial_mat_a(p_mat_a_f, adj_mat, re_root_l[i][0],  re_root_l[i][1])
    #     xw(x_npy, w_npy, re_root_l[i][0],  re_root_l[i][1])

    print(f'\n Dataset: {dataset}:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    num_classes = dataset.num_classes

    # Get the first graph object.
    data = dataset[0]
    print(data)
    # x = data.x.numpy()
    # print(f'Feature vector. Shape {x.shape}')
    # np.save('feature_vector.npy', x)

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    # print(f'Number of training nodes: {data.train_mask.sum()}')
    # print(f'Number of test nodes: {data.test_mask.sum()}')

    # show_adj0(data)

    graph_x = to_networkx(data, to_undirected=True)

    # gen_metis_input(graph_x, data.num_nodes, data.num_edges, './g_d0/Reddit.txt')
    # traversal_order, root_l = gpmetis_clustering('./g_d0/Reddit.txt.part.40', data.num_nodes)
    # graph, re_root_l = reordering1(graph_x, traversal_order, root_l)
    # adj_mat = show_adj2(graph, re_root_l, 8)
    # base_name = './mat/Reddit_part'
    # save_partitions(graph, re_root_l, base_name)

    # p_adj = np.load('./mat/Reddit_part20.npy')
    # rows, cols = p_adj.shape
    # file_name = './mat/Reddit_part0_nnz.txt'
    # f = open(file_name, 'w')
    # total_nnz = 0
    # for i in range(rows):
    #     nnz = np.count_nonzero(p_adj[i, :])
    #     if i > 0 and i % 8 == 0:
    #         f.write('\n')
    #     else:
    #         f.write(f'{nnz} ')
    #     total_nnz += nnz
    # print(f'density {total_nnz / (rows * rows)}')

    # cmap = ListedColormap(['w', 'k'])
    # plt.matshow(p_adj, cmap=cmap)
    # plt.colorbar()
    # plt.show()

    # gen_metis_input(graph_x, data.num_nodes, data.num_edges, './g_d0/cora.txt')
    # traversal_order, root_l = gpmetis_clustering('./g_d0/cora.txt.part.16', data.num_nodes)
    # graph, re_root_l = reordering1(graph_x, traversal_order, root_l)
    # adj_mat = show_adj2(graph, re_root_l, 1)

    # visualize0(graph_x, color=data.y.tolist())
    # gen_nse(graph_x, './g_d0/cora.nse')
    # traversal_order, root_l = pscan_clustering('./g_d0/pscan_clustering_cora.txt', data.num_nodes)
    # graph, re_root_l = reordering0(graph_x, traversal_order, root_l)
    # adj_mat = show_adj1(graph, re_root_l)

    num_classes = 14
    traversal_order, root_l = metis_partition(graph_x, num_classes)
    graph, re_root_l = reordering0(graph_x, traversal_order, root_l)
    adj_mat = show_adj1(graph, re_root_l)

    # traversal_order, root_l = louvain_clustering(graph_x, data.y.tolist())
    # graph, re_root_l = reordering0(graph_x, traversal_order, root_l)
    # adj_mat = show_adj1(graph, re_root_l)

    # traversal_order = degree_sort(graph_x)

    # traversal_order = bdfs(graph_x, 40)
    # reordering0(graph_x, traversal_order)

    # traversal_order, root_l = bbfs(graph_x, 40)
    # graph, re_root_l = reordering0(graph_x, traversal_order, root_l)
    # adj_mat = show_adj1(graph, re_root_l)
    # x_npy = './feature_vector.npy'
    # w_npy = './conv1_lin_weight.npy'
    # x = np.load(x_npy)
    # w = np.load(w_npy)
    # b = np.dot(x, w.transpose())
    # ab(adj_mat, b, re_root_l)

    # traversal_order = degree_bfs(graph_x, 5, 0.1)
    # reordering0(graph_x, traversal_order)

    model = GCN(hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # for epoch in range(1, 110):
    #     loss = train(model, data)
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # print('named_modules')
    # for name, module in model.named_modules():
    #     print(name, module)
    #
    # print('named_parameters')
    # for name, parameter in model.named_parameters():
    #     print(name, parameter)
    #
    # test_acc = test(model, data)

    # with torch.no_grad():
    #     for name, parameter in model.named_parameters():
    #         if name == 'conv1.lin.weight':
    #             para = parameter.detach().clone().numpy()
    #             shape = para.shape
    #             print(name, shape)
    #             np.save('conv1_lin_weight.npy', para)

    # with torch.no_grad():
    #     for name, parameter in model.named_parameters():
    #         para = parameter.detach().clone().numpy()
    #         num_dim = para.ndim
    #         shape = para.shape
    #         bin_l = []
    #
    #         if num_dim == 1:
    #             for p in para:
    #                 para_bin = float_to_bin(p)
    #                 bin_l.append(para_bin)
    #         elif num_dim == 2:
    #             for i in range(shape[0]):
    #                 for j in range(shape[1]):
    #                     para_bin = float_to_bin(para[i][j])
    #                     bin_l.append(para_bin)
    #
    #         # para_bin = np.float16(para[0]).tobytes()
    #         # para_bin = float2bin(para[0])
    #         # para_bin = np.binary_repr(para[0])
    #
    #         print('\n binary parameter: ')
    #         print(bin_l)

    # model.eval()
    # out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)

    # visualize0(graph_x, color=pred.tolist())
    # louvain_clustering(graph_x, pred.tolist())
    # visualize1(out, color=data.y)
