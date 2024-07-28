import time
import sys
import os
import math
import torch as t
from torch import nn
import torch.nn.functional as F
import dgl
from transformers import BertModel, BertTokenizer
from tqdm.auto import tqdm
import numpy as np
import gzip
import pickle
from config import DefaultConfig


sys.path.append("../")
configs = DefaultConfig()

dim_64 = 128
USE_EFEATS = True


class BlockLayer(nn.Module):
    def __init__(self, nfeats_in_dim, nfeats_out_dim, edge_dim=2, use_efeats=USE_EFEATS):
        super(BlockLayer, self).__init__()
        self.use_efeats = use_efeats
        if self.use_efeats:
            self.attn_fc = nn.Linear(2 * nfeats_out_dim + edge_dim, 1, bias=False)
            self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
            self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        else:
            self.attn_fc = nn.Linear(2 * nfeats_out_dim, 1, bias=False)
        self.weight = nn.Parameter(t.Tensor(1, nfeats_in_dim, nfeats_out_dim))
        self.weight2 = nn.Parameter(t.Tensor(1, nfeats_in_dim, nfeats_out_dim))
        self.weight3 = nn.Parameter(t.Tensor(1, nfeats_in_dim, nfeats_out_dim))
        self.bias = nn.Parameter(t.Tensor(1, nfeats_out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.use_efeats:
            nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)

    def edge_attention(self, edges):
        if self.use_efeats:
            z2 = t.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
            a = self.attn_fc(z2)
        else:
            z2 = t.cat([edges.src['z'], edges.dst['z']], dim=1)
            a = self.attn_fc(z2)

        if self.use_efeats:
            ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])
            return {'e': F.leaky_relu(a), 'ez': ez}
        else:
            return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        if self.use_efeats:
            return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}
        else:
            return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = t.sum(attn_w * nodes.mailbox['z'], dim=1)
        if self.use_efeats:
            h = h + t.sum(attn_w * nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, h, e, adj):
        z = h
        L = self.get_laplacian(adj)
        N = adj.shape[0]
        w = self.heat_weight(L, N)
        result1 = t.matmul(w[0], z)
        result2 = t.matmul(w[1], z)
        result3 = t.matmul(w[2], z)
        result = t.matmul(result1, self.weight) + t.matmul(result2, self.weight2) + t.matmul(result3, self.weight3)
        z = t.sum(result, dim=0) + self.bias
        g.ndata['z'] = z
        if self.use_efeats:
            ex = self.fc_edge_for_att_calc(e)
            g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')

    def weight_wavelet(self, s, lamb, U, k, N):
        if t.torch.cuda.is_available():
            multi_order_laplacian = t.zeros([k, N, N]).cuda()
            multi_order_laplacian[0] = t.eye(N).cuda()
        else:
            multi_order_laplacian = t.zeros([k, N, N])
            multi_order_laplacian[0] = t.eye(N)
        for m in range(1, k):
            for i in range(len(lamb)):
                lamb[i] = math.pow(math.e, -lamb[i] * s * m)
            Weight = t.mm(t.mm(U, t.diag(lamb)), t.transpose(U, 0, 1))
            multi_order_laplacian[m] = Weight

        return multi_order_laplacian


    def sort(self, lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]


    def heat_weight(self, laplacian, N):
        lamb, U = t.linalg.eigh(laplacian)
        lamb, U = self.sort(lamb, U)
        weight = self.weight_wavelet(2, lamb, U, 3, N)
        return weight


    @staticmethod
    def get_laplacian(graph):
        D = t.diag(t.sum(graph, dim=-1) ** (-1 / 2))
        if t.torch.cuda.is_available():
            L = t.eye(graph.size(0)).cuda() - t.mm(t.mm(D, graph), D)
        else:
            L = t.eye(graph.size(0)) - t.mm(t.mm(D, graph), D)

        return L


class CoGAT(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = 0.7
        self.graph_lambda = 1.5
        if t.torch.cuda.is_available():
            self.beta_1 = t.log(t.FloatTensor([self.graph_lambda / 1 + 1])).cuda()
            self.beta_2 = t.log(t.FloatTensor([self.graph_lambda / 2 + 1])).cuda()
            self.beta_3 = t.log(t.FloatTensor([self.graph_lambda / 3 + 1])).cuda()
            self.beta_4 = t.log(t.FloatTensor([self.graph_lambda / 4 + 1])).cuda()
            self.beta_5 = t.log(t.FloatTensor([self.graph_lambda / 5 + 1])).cuda()
            self.beta_6 = t.log(t.FloatTensor([self.graph_lambda / 6 + 1])).cuda()
            self.beta_7 = t.log(t.FloatTensor([self.graph_lambda / 7 + 1])).cuda()
        else:
            self.beta_1 = t.log(t.FloatTensor([self.graph_lambda / 1 + 1]))
            self.beta_2 = t.log(t.FloatTensor([self.graph_lambda / 2 + 1]))
            self.beta_3 = t.log(t.FloatTensor([self.graph_lambda / 3 + 1]))
            self.beta_4 = t.log(t.FloatTensor([self.graph_lambda / 4 + 1]))
            self.beta_5 = t.log(t.FloatTensor([self.graph_lambda / 5 + 1]))
            self.beta_6 = t.log(t.FloatTensor([self.graph_lambda / 6 + 1]))
            self.beta_7 = t.log(t.FloatTensor([self.graph_lambda / 7 + 1]))

        self.w = nn.Linear(configs.protBERT_dim + configs.position_encoding_dim, dim_64)
        self.w1 = nn.Parameter(t.randn(dim_64, dim_64))
        self.w2 = nn.Parameter(t.randn(dim_64, dim_64))
        self.w3 = nn.Parameter(t.randn(dim_64, dim_64))
        self.w4 = nn.Parameter(t.randn(dim_64, dim_64))
        self.w5 = nn.Parameter(t.randn(dim_64, dim_64))
        self.w6 = nn.Parameter(t.randn(dim_64, dim_64))
        self.w7 = nn.Parameter(t.randn(dim_64, dim_64))

        self.coevo1 = BlockLayer(128, 128, 1, True)
        self.coevo2 = BlockLayer(128, 128, 1, True)
        self.coevo3 = BlockLayer(128, 128, 1, True)
        self.coevo4 = BlockLayer(128, 128, 1, True)
        self.coevo5 = BlockLayer(128, 128, 1, True)
        self.coevo6 = BlockLayer(128, 128, 1, True)
        self.coevo7 = BlockLayer(128, 128, 1, True)
        
        self.graph_dropout_1 = nn.Dropout(p=configs.graph_dropout_rate)
        self.graph_dropout_2 = nn.Dropout(p=configs.graph_dropout_rate)
        self.graph_dropout_3 = nn.Dropout(p=configs.graph_dropout_rate)
        self.graph_dropout_4 = nn.Dropout(p=configs.graph_dropout_rate)
        self.graph_dropout_5 = nn.Dropout(p=configs.graph_dropout_rate)
        self.graph_dropout_6 = nn.Dropout(p=configs.graph_dropout_rate)
        self.graph_dropout_7 = nn.Dropout(p=configs.graph_dropout_rate)

        self.r_alpha = 0.5
        if t.torch.cuda.is_available():
            self.r0 = t.FloatTensor([self.r_alpha]).cuda()
            self.r1 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 1)]).cuda()
            self.r2 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 2)]).cuda()
            self.r3 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 3)]).cuda()
            self.r4 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 4)]).cuda()
            self.r5 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 5)]).cuda()
            self.r6 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 6)]).cuda()
            self.r7 = t.FloatTensor([(1 - self.r_alpha) ** 7]).cuda()
            self.graph_eye = t.eye(dim_64).cuda()
        else:
            self.r0 = t.FloatTensor([self.r_alpha])
            self.r1 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 1)])
            self.r2 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 2)])
            self.r3 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 3)])
            self.r4 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 4)])
            self.r5 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 5)])
            self.r6 = t.FloatTensor([self.r_alpha * ((1 - self.r_alpha) ** 6)])
            self.r7 = t.FloatTensor([(1 - self.r_alpha) ** 7])
            self.graph_eye = t.eye(dim_64)
        
        self.relu = nn.ReLU()

    def forward(self, A, features, msa_features, adj_features, batch_idx=-1, is_test=0):

        h0 = self.w(features)
        h1 = t.mm((1 - self.alpha) * self.coevo1(A, h0, A.edata['ex'], adj_features) + self.alpha * h0, (1 - self.beta_1) * self.graph_eye + self.beta_1 * self.w1)
        h1 = self.graph_dropout_1(self.relu(h1))
        h2 = t.mm((1 - self.alpha) * self.coevo2(A, h1, A.edata['ex'], adj_features) + self.alpha * h0, (1 - self.beta_2) * self.graph_eye + self.beta_2 * self.w2)
        h2 = self.graph_dropout_2(self.relu(h2))
        h3 = t.mm((1 - self.alpha) * self.coevo3(A, h2, A.edata['ex'], adj_features) + self.alpha * h0, (1 - self.beta_3) * self.graph_eye + self.beta_3 * self.w3)
        h3 = self.graph_dropout_3(self.relu(h3))
        h4 = t.mm((1 - self.alpha) * self.coevo4(A, h3, A.edata['ex'], adj_features) + self.alpha * h0, (1 - self.beta_4) * self.graph_eye + self.beta_4 * self.w4)
        h4 = self.graph_dropout_4(self.relu(h4))
        h5 = t.mm((1 - self.alpha) * self.coevo5(A, h4, A.edata['ex'], adj_features) + self.alpha * h0, (1 - self.beta_5) * self.graph_eye + self.beta_5 * self.w5)
        h5 = self.graph_dropout_5(self.relu(h5))
        h6 = t.mm((1 - self.alpha) * self.coevo6(A, h5, A.edata['ex'], adj_features) + self.alpha * h0, (1 - self.beta_6) * self.graph_eye + self.beta_6 * self.w6)
        h6 = self.graph_dropout_6(self.relu(h6))
        h7 = t.mm((1 - self.alpha) * self.coevo7(A, h6, A.edata['ex'], adj_features) + self.alpha * h0, (1 - self.beta_7) * self.graph_eye + self.beta_7 * self.w7)
        h7 = self.graph_dropout_7(self.relu(h7))

        return h7


class PPIModel(nn.Module):
    def __init__(self, class_nums):
        super(PPIModel,self).__init__()
        global configs
        self.path = './'
        self.graph = CoGAT()
        input_dim = dim_64
        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("DNN_layer1", nn.Linear(input_dim, 1))
        self.outLayer = nn.Sequential(nn.Sigmoid())
        self.dropout_1 = nn.Dropout(p=configs.graph_dropout_rate)
    
    def forward(self, coevolution_features, graph_features, raw_features, position_encoding, protein_adj_features, batch_idx=-1, is_test=0):
        features3 = t.cat((raw_features, position_encoding), -1)
        coevolution_features = coevolution_features.squeeze(0)
        features3 = features3.squeeze(0)
        protein_adj_features = protein_adj_features.squeeze(0)
        features2 = self.graph(graph_features, features3, coevolution_features, protein_adj_features, batch_idx, is_test)
        features2 = features2.unsqueeze(0)
        features4_1 = self.dropout_1(self.DNN1(features2))
        features4 = self.outLayer(features4_1)
        return features4

    def load(self,path):
    
        self.load_state_dict(t.load(path))
        
    def save(self,name=None):
        
        if name is None:
            prefix = ""
            name = time.strftime("%y%m%d_%H:%M:%S.pth".format(prefix))
            
        t.save(self.state_dict(),name)
        return name

