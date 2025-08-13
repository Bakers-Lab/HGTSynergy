import json
import pickle
import pandas as pd
import os

from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from tqdm import tqdm

from parameters.parameters_common import dim_ddi_class
from parameters.parameters_gnn import *
from utils import *

drug_features_file = './data/all_drug_encoded_features.csv'
oneil_drug_features_file = './data/oneil_drug_encoded_features.csv'
ddi_file = './data/all_ddi_classes.csv'
ddi_graph_save_file = './data/graph/ddi_graph.pt'
drugbank_id_to_index_file = './data/graph/drugbank_id_to_index.json'
edge_index_dict_file = './data/graph/edge_index_dict.json'
edge_type_dict_file = './data/graph/edge_type_dict.json'
pretrain_ddi_data_file = './data/ddi_xtrain_ytrain.pkl'


def get_cell_data():
    gene_expr = pd.read_csv('data/oneil_cell_feat.csv', sep=',', header=0, index_col=0)
    return gene_expr


def get_drug_data():
    drug_df = pd.read_csv(oneil_drug_features_file, sep=',', header=0, index_col=0)
    return drug_df



def get_synergy_data(synergy_file):
    synergy_data = pd.read_csv(synergy_file, sep=',', header=0)
    return synergy_data


def get_independence_x_y(synergy_data, drug_data, cell_data):
    pbar = tqdm(total=len(synergy_data))
    x_test = []
    y_test = []

    for index, row in synergy_data.iterrows():
        pbar.set_description('数据集加载进度 ' + str(index))
        pbar.update(1)
        
        comb_name = row['id']
        synergy_score = row['synergy']
        if float(synergy_score) >= 30.0:
            synergy_cls = 1
        else:
            synergy_cls = 0

        drugA_name = comb_name.split('_')[0]
        drugB_name = comb_name.split('_')[1]
        cell_name = comb_name.split('_')[2]

        drugA_features = drug_data.loc[drugA_name][0:].tolist()
        drugB_features = drug_data.loc[drugB_name][0:].tolist()

        cell_features = list(cell_data.loc[cell_name])

        result_1 = {'drugA_features': drugA_features,
                    'drugB_features': drugB_features,
                    'cell_features': cell_features,
                    'drugA_drugbank_id': row['drugA_drugbank_id'],
                    'drugB_drugbank_id': row['drugB_drugbank_id'],
                    'ddi_ids': row['drugA_drugbank_id'] + '-' + row['drugB_drugbank_id']}
        result_2 = {'drugA_features': drugB_features,
                    'drugB_features': drugA_features,
                    'cell_features': cell_features,
                    'drugA_drugbank_id': row['drugA_drugbank_id'],
                    'drugB_drugbank_id': row['drugB_drugbank_id'],
                    'ddi_ids': row['drugA_drugbank_id'] + '-' + row['drugB_drugbank_id']}

        y = {
            'synergy_score': synergy_score,
            'synergy_cls': synergy_cls
        }
        x_test.append(result_1)
        y_test.append(y)

        x_test.append(result_2)
        y_test.append(y)

    pbar.close()
    return x_test, y_test
    

def load_nest_cv_fine_tuning_dataset(test_fold, val_fold):
    all_synergy_data = get_synergy_data(f'./data/cv_labels_all.csv')
    
    
    test_synergy_data = all_synergy_data[all_synergy_data['fold'] == test_fold]
    val_synergy_data = all_synergy_data[all_synergy_data['fold'] == val_fold]
    train_synergy_data = all_synergy_data[(all_synergy_data['fold'] != test_fold) & (all_synergy_data['fold'] != val_fold)]

    drug_data = get_drug_data()
    cell_data = get_cell_data()

    x_train_list, y_train_list = get_independence_x_y(synergy_data=train_synergy_data, drug_data=drug_data, cell_data=cell_data)
    x_val_list, y_val_list = get_independence_x_y(synergy_data=val_synergy_data, drug_data=drug_data, cell_data=cell_data)
    x_test_list, y_test_list = get_independence_x_y(synergy_data=test_synergy_data, drug_data=drug_data, cell_data=cell_data)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list



def load_nest_test_dataset(test_fold):
    synergy_file = f'./data/cv_labels_all.csv'

    all_synergy_data = get_synergy_data(synergy_file=synergy_file)
    outer_train_synergy_data = all_synergy_data[all_synergy_data['fold'] != test_fold]
    test_synergy_data = all_synergy_data[all_synergy_data['fold'] == test_fold]
    
    drug_data = get_drug_data()
    cell_data = get_cell_data()

    x_test_list, y_test_list = get_independence_x_y(synergy_data=test_synergy_data, drug_data=drug_data, cell_data=cell_data)
    x_outer_train_list, y_outer_train_list = get_independence_x_y(synergy_data=outer_train_synergy_data, drug_data=drug_data, cell_data=cell_data)
    return x_outer_train_list, y_outer_train_list, x_test_list, y_test_list


def load_ddi_graph_data(force_reload=False):
    device = get_device()
    if os.path.exists(ddi_graph_save_file) and not force_reload:
        print('之前已经存过处理后的数据，直接从文件中加载！')
        data = torch.load(ddi_graph_save_file, map_location=device)
        with open(drugbank_id_to_index_file, 'r') as f:
            drugbank_id_to_index = json.load(f)
        with open(edge_type_dict_file, 'r') as f:
            edge_type_dict = json.load(f)
        with open(edge_index_dict_file, 'r') as f:
            edge_index_dict = json.load(f)
        
    else:
        print('之前没存过处理后的数据，现在开始处理......')
        drug_features_df = pd.read_csv(drug_features_file)
        drug_id_to_features = {}

        drugbank_id_to_index = {}
        pbar1 = tqdm(total=len(drug_features_df))
        for index, row in drug_features_df.iterrows():
            pbar1.set_description('数据集加载，辅助字典构造进度  ' + str(index))
            pbar1.update(1)
            drugbank_id = row['drugbank_id']
            drug_id_to_features[drugbank_id] = row.iloc[1:].tolist()
            drugbank_id_to_index[drugbank_id] = index

        pbar1.close()

        drug_features_df = drug_features_df.drop(drug_features_df.columns[0], axis=1)
        drug_features_tensor = torch.tensor(drug_features_df.values, dtype=torch.float).to(device)

        ddi_classes_df = pd.read_csv(ddi_file)

        edge_index = []
        ddi_classes_tensor = []

        edge_index_dict = {}
        edge_type_dict = {}
        
        
        pbar2 = tqdm(total=len(ddi_classes_df))
        for index, row in ddi_classes_df.iterrows():
            pbar2.set_description('数据集加载进度  ' + str(index))
            pbar2.update(1)
            drugA_id = row['DrugA-id']
            drugB_id = row['DrugB-id']

            if not (drugA_id in drug_id_to_features and drugB_id in drug_id_to_features):
                print(drugA_id, drugB_id)
            assert drugA_id in drug_id_to_features and drugB_id in drug_id_to_features

            drugA_index = drugbank_id_to_index[drugA_id]
            drugB_index = drugbank_id_to_index[drugB_id]

            edge_index.append([drugA_index, drugB_index])
            ddi_classes_tensor.append([drugA_index, drugB_index, row['ddi_class']])

            if drugA_index not in edge_index_dict:
                edge_index_dict[drugA_index] = []
            
            edge_index_dict[drugA_index].append([drugA_index, drugB_index])
            edge_type_dict[str(drugA_index) + '-' + str(drugB_index)] = row['ddi_class']
            
        pbar2.close()

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        ddi_classes_tensor = torch.tensor(ddi_classes_tensor, dtype=torch.long).t().contiguous().to(device)

        data = Data(x=drug_features_tensor, edge_index=edge_index, edge_type=ddi_classes_tensor)

        torch.save(data, ddi_graph_save_file)
        with open(drugbank_id_to_index_file, 'w') as f:
            json.dump(drugbank_id_to_index, f)
        with open(edge_type_dict_file, 'w') as f:
            json.dump(edge_type_dict, f)
        with open(edge_index_dict_file, 'w') as f:
            json.dump(edge_index_dict, f)
        print('图对象已经保存在文件中！')

    ddi_graph_data = Data(
        x=data.x,
        edge_index=data.edge_index,
        edge_type=data.edge_type[2, :]
    ).to(device)
    return ddi_graph_data, drugbank_id_to_index, edge_index_dict, edge_type_dict

    
def batch_edges(graph: Data, batch_size):
    edge_index = graph.edge_index.t().tolist()
    edge_type = graph.edge_type.tolist()
    num_edges = len(edge_index)
    
    # 分批处理
    batches = []
    for i in range(0, num_edges, batch_size):
        edge_batch = edge_index[i:i + batch_size]
        type_batch = edge_type[i:i + batch_size]
        batches.append((edge_batch, type_batch))
    
    return batches


def load_pre_train_dataset():
    if os.path.exists(pretrain_ddi_data_file):
        print('几百万条ddi数据之前已经存储过了，现在直接从文件中读取')
        with open(pretrain_ddi_data_file, 'rb') as f:
            loaded_list = pickle.load(f)
        x_train = loaded_list[0]
        y_train = loaded_list[1]
        return x_train, y_train

    print('几百万条ddi数据之前没有存储过，现在现场计算')
    print('开始加载pretrain数据...')
    ddi_df = pd.read_csv(ddi_file, header=0)
    all_drug_features_df = pd.read_csv(drug_features_file, header=0)

    x_train = []
    y_train = []

    drug_id_to_features = {}
    pbar1 = tqdm(total=len(all_drug_features_df))
    for index, row in all_drug_features_df.iterrows():
        pbar1.set_description('数据集加载，辅助字典构造进度  ' + str(index))
        pbar1.update(1)
        drug_id_to_features[row['drugbank_id']] = row.iloc[1:].tolist()
    pbar1.close()

    pbar2 = tqdm(total=len(ddi_df))
    for index, row in ddi_df.iterrows():
        pbar2.set_description('数据集加载进度 ' + str(index))
        pbar2.update(1)

        if row['DrugA-id'] in drug_id_to_features and row['DrugB-id'] in drug_id_to_features:
            x_train.append({
                'drugA_features': drug_id_to_features[row['DrugA-id']],
                'drugB_features': drug_id_to_features[row['DrugB-id']]
            })
            y_train.append(row['ddi_class'])
    pbar2.close()

    with open(pretrain_ddi_data_file, 'wb') as f:
        pickle.dump([x_train, y_train], f)

    print('pretrain数据加载完成！')
    return x_train, y_train


def split_into_train_val(x_train, y_train, split_rate):
    combined = list(zip(x_train, y_train))

    random.shuffle(combined)

    new_train_size = int(len(combined) * split_rate)

    x_train_new, y_train_new = zip(*combined[:new_train_size])
    x_val, y_val = zip(*combined[new_train_size:])
    
    return list(x_train_new), list(x_val), list(y_train_new), list(y_val)


def get_heterogeneous_data(data: Data):
    edge_index = data.edge_index
    edge_type = data.edge_type

    edge_index_dict = {}

    num_edges = edge_index.size(1)
    for i in range(dim_ddi_class):
        mask = (edge_type == i).nonzero(as_tuple=False).view(-1)
        if mask.numel() > 0:
            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            current_edge_index = torch.stack([src, dst], dim=0)
            edge_index_dict[('drug', f'edge_type_{i}', 'drug')] = current_edge_index

    data.edge_index_dict = edge_index_dict
    return data


def get_fine_tuning_loaders(x_train, y_train, x_val, y_val, x_test, y_test, drugbank_id_to_index=None, edge_type_dict=None):
    class TensorDatasetWithIndexList(TensorDataset):
        def __init__(self, *tensors, ddi_index_tensor):
            self.tensors = tensors
            self.ddi_index_tensor = ddi_index_tensor
            assert all(tensors[0].size(0) == tensor.size(0) == self.ddi_index_tensor.size(0) for tensor in tensors)

        def __len__(self):
            return self.tensors[0].size(0)

        def __getitem__(self, index):
            return tuple(tensor[index] for tensor in self.tensors), self.ddi_index_tensor[index]
    
    def get_index_tensor(x_list):
        ddi_index_list_train = []
        for item in x_list: 
            id_list = item["ddi_ids"].split("-")

            drugA_id = id_list[0]
            drugB_id = id_list[1]

            drugA_index = int(drugbank_id_to_index[drugA_id])
            drugB_index = int(drugbank_id_to_index[drugB_id])

            if edge_type_dict:
                if f"{drugA_index}-{drugB_index}" in edge_type_dict:
                    ddi_class = int(edge_type_dict[f"{drugA_index}-{drugB_index}"])
                else:
                    ddi_class = -1

                ddi_index_list_train.append([drugA_index, drugB_index, ddi_class])
            else:
                ddi_index_list_train.append([drugA_index, drugB_index])
        
        ddi_index_tensor_train = torch.tensor(ddi_index_list_train, dtype=torch.long).to(get_device())
        return ddi_index_tensor_train


    device = get_device()

    drugA_features_tensor_train = torch.Tensor([sample['drugA_features'] for sample in x_train]).to(device)
    drugB_features_tensor_train = torch.Tensor([sample['drugB_features'] for sample in x_train]).to(device)
    cell_features_tensor_train = torch.Tensor([sample['cell_features'] for sample in x_train]).to(device)
    y_train_regression_tensor = torch.Tensor([float(sample['synergy_score']) for sample in y_train]).to(device)
    y_train_cls_tensor = torch.Tensor([float(sample['synergy_cls']) for sample in y_train]).to(device)

    drugA_features_tensor_val = torch.Tensor([sample['drugA_features'] for sample in x_val]).to(device)
    drugB_features_tensor_val = torch.Tensor([sample['drugB_features'] for sample in x_val]).to(device)
    cell_features_tensor_val = torch.Tensor([sample['cell_features'] for sample in x_val]).to(device)
    y_val_regression_tensor = torch.Tensor([float(sample['synergy_score']) for sample in y_val]).to(device)
    y_val_cls_tensor = torch.Tensor([float(sample['synergy_cls']) for sample in y_val]).to(device)

    drugA_features_tensor_test = torch.Tensor([sample['drugA_features'] for sample in x_test]).to(device)
    drugB_features_tensor_test = torch.Tensor([sample['drugB_features'] for sample in x_test]).to(device)
    cell_features_tensor_test = torch.Tensor([sample['cell_features'] for sample in x_test]).to(device)
    y_test_regression_tensor = torch.Tensor([float(sample['synergy_score']) for sample in y_test]).to(device)
    y_test_cls_tensor = torch.Tensor([float(sample['synergy_cls']) for sample in y_test]).to(device)


    ddi_index_tensor_train = get_index_tensor(x_train)
    ddi_index_tensor_val = get_index_tensor(x_val)
    ddi_index_tensor_test = get_index_tensor(x_test)

    batch_size = gnn_finetuning_batch_size
    train_dataset = TensorDatasetWithIndexList(drugA_features_tensor_train, drugB_features_tensor_train, cell_features_tensor_train, y_train_regression_tensor, y_train_cls_tensor, ddi_index_tensor=ddi_index_tensor_train)
    val_dataset = TensorDatasetWithIndexList(drugA_features_tensor_val, drugB_features_tensor_val, cell_features_tensor_val, y_val_regression_tensor, y_val_cls_tensor, ddi_index_tensor=ddi_index_tensor_val)
    test_dataset = TensorDatasetWithIndexList(drugA_features_tensor_test, drugB_features_tensor_test, cell_features_tensor_test, y_test_regression_tensor, y_test_cls_tensor, ddi_index_tensor=ddi_index_tensor_test)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader
