import pickle
import numpy as np
import torch
import dgl


def cal_edges(adjacency_matrix_input):  # to get the index of the edges
    mask = adjacency_matrix_input > 1e-5
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list


def cal_edge_attr(index_list, adjacency_matrix_input):
    distance = []
    for i in range(len(index_list[0])):
        cur_s = index_list[0][i]
        cur_t = index_list[1][i]
        cur_dist = adjacency_matrix_input[cur_s][cur_t]
        distance.append(cur_dist)
    radius_attr_list = np.array([distance])
    return radius_attr_list


def add_edges_custom(G, radius_index_list, edge_features):
    src, dst = radius_index_list[1], radius_index_list[0]
    G.add_edges(src, dst)
    G.edata['ex'] = torch.tensor(edge_features)


def generate_graph(dataset_name, graph_data_features):
    dgl_graph_features = []
    for i in range(len(graph_data_features)):
        cur_adj_mat = graph_data_features[i]
        cur_adj_mat = np.array(cur_adj_mat)
        cur_radius_index_list = cal_edges(cur_adj_mat)
        edge_feat = cal_edge_attr(cur_radius_index_list, cur_adj_mat)
        G = dgl.DGLGraph()
        G.add_nodes(cur_adj_mat.shape[0])
        edge_feat = np.transpose(edge_feat)
        add_edges_custom(G, cur_radius_index_list, edge_feat)
        dgl_graph_features.append(G)
    pkl_file = open('features/' + dataset_name + '_dgl_graphs.pkl', 'wb')
    pickle.dump(dgl_graph_features, pkl_file)
    pkl_file.close()
