import pickle
import numpy as np
from torch.utils import data
from config import DefaultConfig


configs = DefaultConfig()


class dataSet(data.Dataset):
    def __init__(self, label_file=None, protein_list_file=None, train_MSA_file=None, train_graph_file=None, train_protBERT_file=None, train_position_encoding_file=None, train_adj_file=None):
        super(dataSet,self).__init__()

        self.all_label = []
        for lab_file in label_file:
            with open(lab_file, "rb") as fp_label:
                temp_label = pickle.load(fp_label)
            self.all_label.extend(temp_label)

        self.protein_list = []
        for list_file in protein_list_file:
            with open(list_file, "rb") as list_label:
                temp_list = pickle.load(list_label)
            self.protein_list.extend(temp_list)

        self.MSA_features_matrix = []
        for dca_file in train_MSA_file:
            with open(dca_file, "rb") as fp_dca:
                temp_dca = pickle.load(fp_dca)
            self.MSA_features_matrix.extend(temp_dca)

        self.graph_matrix = []
        for graph_file in train_graph_file:
            with open(graph_file, "rb") as fp_graph:
                temp_graph = pickle.load(fp_graph)
            self.graph_matrix.extend(temp_graph)

        self.all_protBERT = []
        for protBERT_file in train_protBERT_file:
            with open(protBERT_file, "rb") as fp_protBERT:
                temp_protBERT = pickle.load(fp_protBERT)
            self.all_protBERT.extend(temp_protBERT)

        self.all_position_encoding = []
        for position_encoding_file in train_position_encoding_file:
            with open(position_encoding_file, "rb") as fp_position_encoding:
                temp_position_encoding = pickle.load(fp_position_encoding)
            self.all_position_encoding.extend(temp_position_encoding)

        self.adj_matrix = []
        for adj_file in train_adj_file:
            with open(adj_file, "rb") as fp_adj:
                temp_adj = pickle.load(fp_adj)
            self.adj_matrix.extend(temp_adj)


    def __getitem__(self, index):
        protein_coevolution = np.array(self.MSA_features_matrix[index])
        protein_graph = self.graph_matrix[index]
        protein_protBERT = np.array(self.all_protBERT[index])
        protein_position_encoding = np.array(self.all_position_encoding[index])
        protein_adj = np.array(self.adj_matrix[index])
        protein_label = np.array(self.all_label[index])
        return protein_coevolution, protein_graph, protein_protBERT, protein_position_encoding, protein_adj, protein_label

    def __len__(self):
            return len(self.protein_list)
