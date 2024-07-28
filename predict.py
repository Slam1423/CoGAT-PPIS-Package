import os
import pickle
import numpy as np
import torch
import torch.utils.data.sampler as sampler
import dgl
from config import DefaultConfig
import data_generator
from ppi_model_CoGAT import PPIModel


configs = DefaultConfig()


def graph_collate(samples):
    protein_coevolution, G, protein_protBERT, protein_position_encoding, protein_adj, protein_label = map(list, zip(*samples))
    protein_label = torch.Tensor(protein_label)
    G_batch = dgl.batch(G)
    protein_protBERT = torch.Tensor(protein_protBERT)
    protein_coevolution = torch.Tensor(protein_coevolution)
    protein_position_encoding = torch.Tensor(protein_position_encoding)
    protein_adj = torch.Tensor(protein_adj)
    return protein_coevolution, G_batch, protein_protBERT, protein_position_encoding, protein_adj, protein_label


def test(model, loader, path_dir, threshold, test_name):
    model.eval()
    result = []
    all_trues = []
    for batch_idx, (coevolution_data, graph_data, protBERT_data, position_encoding_data, adj_data, label) in enumerate(loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                msa_var = torch.autograd.Variable(coevolution_data.cuda().float())
                graph_data.edata['ex'] = torch.autograd.Variable(graph_data.edata['ex'].float())
                graph_data = graph_data.to(torch.device('cuda:0'))
                protBERT_var = torch.autograd.Variable(protBERT_data.cuda().float())
                position_encoding_var = torch.autograd.Variable(position_encoding_data.cuda().float())
                adj_data_var = torch.autograd.Variable(adj_data.cuda().float())
                label_var = torch.autograd.Variable(label.cuda().float())
            else:
                msa_var = torch.autograd.Variable(coevolution_data.float())
                graph_data.edata['ex'] = torch.autograd.Variable(graph_data.edata['ex'].float())
                protBERT_var = torch.autograd.Variable(protBERT_data.float())
                position_encoding_var = torch.autograd.Variable(position_encoding_data.float())
                adj_data_var = torch.autograd.Variable(adj_data.float())
                label_var = torch.autograd.Variable(label.float())
        if torch.cuda.is_available():
            batch_idx_var = torch.tensor(batch_idx).cuda().float()
            is_test_var = torch.tensor([1.0]).cuda()
        else:
            batch_idx_var = torch.tensor(batch_idx).float()
            is_test_var = torch.tensor([1.0])
        output = model(msa_var, graph_data, protBERT_var, position_encoding_var, adj_data_var, batch_idx_var, is_test_var)
        shapes = output.data.shape
        output = output.view(shapes[0] * shapes[1])
        result.append(output.data.cpu().numpy())
        all_trues.append(label.numpy().squeeze(0))

    # caculate
    pred_list = []
    for i in range(len(result)):
        pred = result[i] >= threshold
        pred_list.append(pred.astype(int).tolist())
    result_file = "predict_result_dir/" + test_name + "_predict_result.pkl"
    with open(result_file, "wb") as fp:
        pickle.dump(pred_list, fp)
    print('prediction done.')


def predict(model_file, test_data, path_dir, threshold=0.5, test_name=''):
    test_label_file = ['features/{0}_fake_label.pkl'.format(key) for key in test_data]
    test_graph_file = ['features/{0}_dgl_graphs.pkl'.format(key) for key in test_data]
    test_adj_file = ['features/{0}_ones_graphs_adj.pkl'.format(key) for key in test_data]
    test_protBERT_file = ['features/{0}_raw_features.pkl'.format(key) for key in test_data]
    test_MSA_file = ['features/{0}_coevolutionary_features.pkl'.format(key) for key in test_data]
    test_position_encoding_file = ['features/{0}_position_encoding_length.pkl'.format(key) for key in test_data]

    test_list_file = 'features/' + test_name + '_predict_list.pkl'
    all_list_file = 'features/' + test_name + '_all_protein.pkl'

    batch_size = configs.batch_size
    all_list_file = [all_list_file]
    test_dataSet = data_generator.dataSet(test_label_file, all_list_file, test_MSA_file, test_graph_file, test_protBERT_file, test_position_encoding_file, test_adj_file)
    with open(test_list_file,"rb") as fp:
        test_list = pickle.load(fp)
    test_samples = sampler.SequentialSampler(test_list)
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=batch_size, sampler=test_samples, pin_memory=True, num_workers=0, drop_last=False, collate_fn=graph_collate)
    # Models
    class_nums = 1
    model = PPIModel(class_nums)
    if torch.torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
        model = model.cuda()
        model.eval()
    else:
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        model.eval()
    return test(model, test_loader, path_dir, threshold, test_name)


def make_prediction(model_file_name, dataset, threshold=0.50, test_name=''):
    path_dir = "./"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    print('Start predicting...')
    predict(model_file_name, dataset, path_dir, threshold, test_name)
