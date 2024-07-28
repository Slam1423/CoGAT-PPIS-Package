import sys
from Bio.Blast import NCBIWWW
import biolib
import pickle
import numpy as np
import pandas as pd
import re
from plmDCA import plmDCA_main
from predict import make_prediction
from ProtBERT_feature_generator.ProtBERT_feature_generator import calc_ProtBERT
from graph_generator import generate_graph
import signal
import subprocess
from normalize import feature_normalize


class Timeout(Exception):
   pass

def handler(sig, frame):
   raise Timeout


dataset_name = sys.argv[2]
database = sys.argv[4]
hitlist_size = int(sys.argv[6])
input_file = 'raw_input_sequences/' + dataset_name + '_seq.txt'

signal.signal(signal.SIGALRM, handler)

# raw sequence -> MSA files
file = open(input_file, 'r')
target_protein_file_list = file.readlines()
file.close()

blast_num = 20
blast_protein_num = 10

target_len_list = []
target_protein_list = []
target_protein_name_list = []
target_len_list1 = []
target_protein_list1 = []
target_protein_name_list1 = []
file = open(input_file, 'r')
target_file_list = file.readlines()
file.close()
for i in range(len(target_file_list)):
   if '>' in target_file_list[i]:
       target_protein_name_list1.append(target_file_list[i][1:-1])
       continue
   target_len_list1.append(len(target_file_list[i][:-1]))
   target_protein_list1.append(target_file_list[i][:-1])
ProtBERT_list = calc_ProtBERT(dataset_name, target_protein_list1)

d_model = 128
position_encoding = []
for i in range(len(target_protein_list1)):
    protein_seq = target_protein_list1[i]
    protein_position_encoding = []
    for pos in range(len(protein_seq)):
        residue_pssm = protein_seq[pos]
        residue_position_encoding = []
        for k in range(d_model):
            if k % 2 == 0:
                residue_position_encoding.append(np.sin(pos / 10000 ** (k / d_model)))
            else:
                residue_position_encoding.append(np.cos(pos / 10000 ** ((k - 1) / d_model)))
        residue_position_encoding.append(len(protein_seq) / 2000)
        protein_position_encoding.append(residue_position_encoding)
    position_encoding.append(protein_position_encoding)

fasta_list = []
for i in range(len(target_protein_file_list)):
   if blast_num * (i+1) < len(target_protein_file_list):
       fasta_list.append(''.join(target_protein_file_list[blast_num*i: blast_num*(i+1)]))
   else:
       fasta_list.append(''.join(target_protein_file_list[blast_num * i:]))
       break

total_cnt = 0
print('Start netsurfp...')
print('biolib.__version__: ' + biolib.__version__)
nsp3 = biolib.load('DTU/NetSurfP-3')
nsp3_results = nsp3.cli(args='-i raw_input_sequences/' + dataset_name + '_seq.txt')
nsp3_results.save_files("biolip_netsurfp/")
netsurf_df = pd.read_csv('biolip_netsurfp/results.csv', header=0)
netsurf_df['sin_phi'] = netsurf_df[' phi'].apply(np.sin)
netsurf_df['cos_phi'] = netsurf_df[' phi'].apply(np.cos)
netsurf_df['sin_psi'] = netsurf_df[' psi'].apply(np.sin)
netsurf_df['cos_psi'] = netsurf_df[' psi'].apply(np.cos)
# sequence_data_list = []
faked_label_list = []
pssm_data_list = []
protein_netsurf_list = []
msa_feature_list = []
quantile_msa_feature_list = []
dset_list = []
print('Start Blastp...')
for iii in range(len(fasta_list)):
   fasta = fasta_list[iii]
   print('Blastp group: ' + str(iii) + ' (Mostly 10 sequences in a group)')
   flag = True
   while flag:
       signal.alarm(7200)
       try:
           result_handle = NCBIWWW.qblast('blastp', database, fasta, alignments=2000, hitlist_size=hitlist_size, format_type="Text")
           res_str = result_handle.read()
           if 'Query' in res_str and 'Sbjct' in res_str and 'ka-blk-sigma' in res_str:
               pass
           else:
               print('continue')
               continue
           file = open('msa_dir/protein_msa_multi_output' + str(iii) + '.txt', 'w')
           file.write(res_str)
           file.close()
           flag = False
       except Exception as e:
           print(e)
       signal.alarm(0)

   # MSA file -> pure MSA
   output_file = open('msa_dir/protein_msa_multi_output' + str(iii) + '.txt', 'r')
   raw_total_list = output_file.read()

   for i in range(len(target_protein_name_list1)):
       name = target_protein_name_list1[i]
       if 'Query= ' + name in raw_total_list:
           target_len_list.append(target_len_list1[i])
           target_protein_list.append(target_protein_list1[i])
           target_protein_name_list.append(name)

   total_list = raw_total_list.split('Query=')[1:]
   for ii in range(len(total_list)):
       each = total_list[ii]
       target_protein = target_protein_list[blast_protein_num * iii + ii]
       curList = each.split('Posted')[0].split('Score')
       length = target_len_list[blast_protein_num * iii + ii]
       msa_list = []
       whole_str_file = ''
       for i in range(len(curList)):
           cur_part = curList[i].split('\n')
           cur_msa_list = ['-' for u in range(length)]
           curcnt = 0
           for line in cur_part:
               if 'Query' in line:
                   lineList = re.split(r"[ ]+", line)
                   s = int(lineList[1])
                   t = int(lineList[-1])
                   curFatherSeq = lineList[2]
               if 'Sbjct' in line:
                   lineList = re.split(r"[ ]+", line)
                   cur_msa = lineList[2]
                   for j in range(len(cur_msa)):
                       if curFatherSeq[j] == '-':
                           continue
                       c_str = cur_msa[j]
                       cur_msa_list[curcnt] = c_str
                       curcnt += 1
           cur_msa_str = ''.join(cur_msa_list)
           msa_list.append(cur_msa_str)
           whole_str_file += cur_msa_str + '\n'
       file = open('msa_dir/protein_pure_msa_output' + str(ii + blast_protein_num * iii) + '.txt', 'w')
       file.write(whole_str_file)
       file.close()

# calculate hmm using hmmbuild
for i in range(len(target_protein_list1)):
    file = open('msa_dir/protein_pure_msa_output' + str(i) + '.txt', 'r')
    content = file.readlines()
    file.close()
    new_str = ''
    for j in range(len(content)):
        new_str += '>' + str(j) + '\n' + content[j]
    file = open('msa_dir/protein_pure_msa_output_formatted_' + str(i) + '.txt', 'w')
    file.write(new_str)
    file.close()
for i in range(len(target_protein_list1)):
    completed_process = subprocess.run('./hmmbuild --symfrac 0 hmm_dir/hmm_' + str(i) + '.txt msa_dir/protein_pure_msa_output_formatted_' + str(i) + '.txt', text=True, shell=True)
    if completed_process.returncode == 0:
        print("subprocess_" + str(i) + ' run successfully.')

hmm_list = []
for i in range(len(target_protein_list1)):
    file = open('hmm_dir/hmm_' + str(i) + '.txt', 'r')
    content = file.readlines()
    file.close()
    cur_protein_hmm = []
    for i in range(len(content)):
        curline = content[i]
        cur_list = curline.split()[1:-5]
        if len(cur_list) == 20:
            cur_list2 = np.array(cur_list).astype(float).tolist()
            cur_protein_hmm.append(cur_list2)
    hmm_list.append(cur_protein_hmm)

# pure MSA -> DCA
print('Start computing DCA, PSSM...')
for ii in range(len(target_protein_name_list)):
   target_protein = target_protein_list[ii]
   plmDCA_main('msa_dir/protein_pure_msa_output' + str(ii) + '.txt', 'msa_dir/protein_pure_dca_output' + str(ii) + '.pkl', 0.1)
   output_file = open('msa_dir/protein_pure_dca_output' + str(ii) + '.pkl', 'rb')
   values = pickle.load(output_file)
   output_file.close()
   protein_seq = target_len_list[ii]
   dca_mat = np.zeros((protein_seq, protein_seq))
   cnt = 0
   for i in range(protein_seq):
       for j in range(i+1, protein_seq):
           dca_mat[i][j] = values[cnt]
           dca_mat[j][i] = values[cnt]
           cnt += 1
   dca_list = []
   for i in range(protein_seq):
        dca_list.append(dca_mat[i].tolist())
   protein_len = target_len_list[ii]
   msa_feature_list.append(dca_list)

   dca_arr = np.array(dca_list)
   dca_quantile = np.percentile(dca_arr, 87)
   dca_arr[dca_arr < dca_quantile] = 0
   quantile_msa_feature_list.append(dca_arr.tolist())
   generate_graph(dataset_name, quantile_msa_feature_list)

   # pure MSA -> PSSM
   print('pure MSA -> PSSM')
   pure_msa_file = open('msa_dir/protein_pure_msa_output' + str(ii) + '.txt', 'r')
   content = pure_msa_file.readlines()
   pure_msa_file.close()
   amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                      'V']
   letter_to_idx = dict()
   for i in range(len(amino_acid_list)):
       letter_to_idx[amino_acid_list[i]] = i
   prime_seq = content[0][:-1]
   theLen = target_len_list[ii]
   matrix = [[0 for u in range(20)] for uu in range(theLen)]
   for i in range(len(content)):
       cur = content[i][:-1]
       for j in range(len(cur)):
           now = cur[j]
           if now not in letter_to_idx:
               continue
           idx = letter_to_idx[now]
           matrix[j][idx] += 1
   for i in range(theLen):
       for j in range(20):
           matrix[i][j] = int(np.log2(20 * (matrix[i][j] + 1) / (len(content) + 1)))
   pssm_data_list.append(matrix)

   # netsurfp
   cur_df = netsurf_df[netsurf_df['id'] == '>' + target_protein_name_list[ii]]
   cur_len = cur_df.shape[0]
   need_cols = [21, 22, 23, 24, 3, 10, 11, 12, 13, 14, 15, 16, 17] # N * 13
   secondary_structure = cur_df.iloc[:, need_cols].values.tolist()
   protein_netsurf_list.append(secondary_structure) # N * 13

   fake_labels = []
   for i in range(theLen):
       fake_labels.append(0)
   faked_label_list.append(fake_labels)

test_list = []
dset_list = []
for i in range(len(protein_netsurf_list)):
   test_list.append(i)
   dset_list.append(i)

protein_netsurf_list = feature_normalize(protein_netsurf_list)
pssm_data_list = feature_normalize(pssm_data_list)
hmm_list = feature_normalize(hmm_list)
ProtBERT_list = feature_normalize(ProtBERT_list)

raw_feature_list = []
for i in range(len(ProtBERT_list)):
    cur_raw = []
    for j in range(len(ProtBERT_list[i])):
        cur_raw.append(protein_netsurf_list[i][j] + pssm_data_list[i][j] + hmm_list[i][j] + ProtBERT_list[i][j].tolist())
    raw_feature_list.append(cur_raw)

pkl_file = open('features/' + dataset_name + '_fake_label.pkl', 'wb')
pickle.dump(faked_label_list, pkl_file)
pkl_file.close()
pkl_file = open('features/' + dataset_name + '_ones_graphs_adj.pkl', 'wb')
pickle.dump(quantile_msa_feature_list, pkl_file)
pkl_file.close()
pkl_file = open('features/' + dataset_name + '_raw_features.pkl', 'wb')
pickle.dump(raw_feature_list, pkl_file)
pkl_file.close()
pkl_file = open('features/' + dataset_name + '_coevolutionary_features.pkl', 'wb')
pickle.dump(msa_feature_list, pkl_file)
pkl_file.close()
pkl_file = open('features/' + dataset_name + '_position_encoding_length.pkl', 'wb')
pickle.dump(position_encoding, pkl_file)
pkl_file.close()
pkl_file = open('features/' + dataset_name + '_predict_list.pkl', 'wb')
pickle.dump(test_list, pkl_file)
pkl_file.close()
pkl_file = open('features/' + dataset_name + '_all_protein.pkl', 'wb')
pickle.dump(dset_list, pkl_file)
pkl_file.close()

make_prediction('trained_PPIModel.dat', [dataset_name], 0.6, dataset_name)
