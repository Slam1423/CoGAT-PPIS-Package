from time import time
import torch
from transformers import BertModel, BertTokenizer
import re
import os
from tqdm.auto import tqdm
import numpy as np
import gzip
import pickle
import pandas as pd


cnn_idx = 13


def generate_protbert_features(fasta_name, seq_list):
    path = './ProtBERT_feature_generator/'
    t0=time()

    vocabFilePath = path+'vocab.txt'
        
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    sequences = seq_list

    sequences_Example = [' '.join(list(seq)) for seq in sequences]
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]

    all_protein_features = []

    for i, seq in enumerate(sequences_Example):
        # print(i)
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=False)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
            
            # print(features.__len__())
        all_protein_features += features

    # pickle.dump(all_protein_features, open('features/' + fasta_name + '_ProtBERT_features.pkl', 'wb'))

    print('Total time spent for ProtBERT:',time()-t0)

    return all_protein_features


def calc_ProtBERT(fasta_name, seq_list):

    return generate_protbert_features(fasta_name, seq_list)
