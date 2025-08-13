import torch
import torch.nn as nn
from torch_geometric.nn import HANConv


class DrugEncoder(nn.Module):
    def __init__(self, dim_drug_features, dim_drug_embs, p_dropout):
        super(DrugEncoder, self).__init__()
        self.fc1 = nn.Linear(dim_drug_features, 1024)
        self.batch1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, dim_drug_embs)
        self.batch2 = nn.BatchNorm1d(dim_drug_embs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, drug_features):
        x_drug = self.fc1(drug_features)
        x_drug = self.relu(x_drug)
        x_drug = self.batch1(x_drug)
        x_drug = self.dropout(x_drug)
        x_drug = self.fc2(x_drug)
        x_drug = self.relu(x_drug)
        x_drug = self.batch2(x_drug)
        return x_drug


class CellEncoder(nn.Module):
    def __init__(self, dim_cell_features, dim_cell_embs, p_dropout):
        super(CellEncoder, self).__init__()
        self.fc1 = nn.Linear(dim_cell_features, 1024)
        self.batch1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, dim_cell_embs)
        self.batch2 = nn.BatchNorm1d(dim_cell_embs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, gexpr_data):
        x_cellline = self.fc1(gexpr_data)     
        x_cellline = self.relu(x_cellline)
        x_cellline = self.batch1(x_cellline)
        x_cellline = self.dropout(x_cellline)
        x_cellline = self.fc2(x_cellline)  
        x_cellline = self.relu(x_cellline)
        x_cellline = self.batch2(x_cellline)
        return x_cellline


class Predictor(nn.Module):
    def __init__(self, dim_drug_embs, dim_cell_embs, dim_ddi_encode, p_dropout, hidden_size):
        super(Predictor, self).__init__()

        self.drug_atten_layer = nn.MultiheadAttention(dim_ddi_encode, num_heads=2, dropout=p_dropout, batch_first=True)
        self.drug_prod_layer = nn.Sequential(
            nn.Linear(dim_drug_embs, dim_ddi_encode)
        )

        self.network = nn.Sequential(
            nn.Linear(2 * dim_drug_embs + dim_cell_embs + 3 * dim_ddi_encode, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drugA_embs, drugB_embs, cell_embs, ddi_embs):
        drugA_ddi_embs = self.drug_prod_layer(drugA_embs)
        drugB_ddi_embs = self.drug_prod_layer(drugB_embs)

        drugA_ddi_embs = self.drug_atten_layer(query=drugA_ddi_embs.unsqueeze(1), key=ddi_embs.unsqueeze(1), value=ddi_embs.unsqueeze(1))[0]
        drugA_ddi_embs = drugA_ddi_embs.squeeze(1)

        drugB_ddi_embs = self.drug_atten_layer(query=drugB_ddi_embs.unsqueeze(1), key=ddi_embs.unsqueeze(1), value=ddi_embs.unsqueeze(1))[0]
        drugB_ddi_embs = drugB_ddi_embs.squeeze(1)

        x_synergy = torch.cat([drugA_embs, drugB_embs, cell_embs, drugA_ddi_embs, drugB_ddi_embs, ddi_embs], dim=1)
        x_synergy = self.network(x_synergy).squeeze()
        return x_synergy


class HGTSynergy(nn.Module):
    def __init__(self, dim_drug_features, dim_drug_embs, dim_cell_features, dim_cell_embs, dim_ddi_encode, dim_hidden, dim_ddi_class, p_dropout, heads, hidden_size, modeling='han'):
        super(HGTSynergy, self).__init__()
        self.drug_encoder = DrugEncoder(dim_drug_features=dim_drug_features, dim_drug_embs=dim_drug_embs, p_dropout=p_dropout)
        self.cell_encoder = CellEncoder(dim_cell_features=dim_cell_features, dim_cell_embs=dim_cell_embs, p_dropout=p_dropout)
        self.ddi_net = HANModel(input_dim=dim_drug_features, hidden_dim=dim_hidden, output_dim=dim_ddi_encode, dim_drug_embs=dim_drug_embs, num_edge_types=dim_ddi_class, p_dropout=p_dropout, heads=heads)
        self.predictor = Predictor(dim_drug_embs=dim_drug_embs, dim_ddi_encode=dim_ddi_encode, dim_cell_embs=dim_cell_embs, p_dropout=p_dropout, hidden_size=hidden_size)

        self.modeling = modeling
        self.dim_ddi_encode = dim_ddi_encode

    def forward(self, drugA_features, drugB_features, cell_features, ddi_graph, ddi_index_tensor):
        edge_list = []
        for ddi in ddi_index_tensor.cpu().numpy():
            drugA_index = ddi[0]
            drugB_index = ddi[1]
            edge_list.append((drugA_index, drugB_index))
            edge_list.append((drugB_index, drugA_index))

        ddi_embs, _, _ = self.ddi_net(ddi_graph, pretrain_start_index=None, pretrain_end_index=None, edge_list=edge_list)
        
        drugA_embs = self.drug_encoder(drugA_features)
        drugB_embs = self.drug_encoder(drugB_features)
        cell_embs = self.cell_encoder(cell_features)

        synergy_score = self.predictor(drugA_embs, drugB_embs, cell_embs, ddi_embs)
        return synergy_score


class MixModel(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout):
        super(MixModel, self).__init__()      
        self.fc1 = nn.Linear(input_dim, input_dim // 2)     
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, output_dim)     
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        return x


class GNNDDIMlp(nn.Module):
    def __init__(self, dim_hidden, dim_ddi_encode, p_dropout):
        super(GNNDDIMlp, self).__init__()
        self.fc1 = nn.Linear(dim_hidden*4, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_ddi_encode)
        self.bn2 = nn.BatchNorm1d(dim_ddi_encode)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        return x


class HANModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dim_drug_embs, num_edge_types, p_dropout, heads):
        super(HANModel, self).__init__()
        self.ddi_atten_layer = nn.MultiheadAttention(2 * hidden_dim, num_heads=2, dropout=p_dropout, batch_first=True)

        self.conv1 = HANConv(
            in_channels={"drug": input_dim},
            out_channels=hidden_dim,
            metadata=(['drug'], [(f'drug', f'edge_type_{i}', 'drug') for i in range(num_edge_types)]),
            heads=heads,
            dropout=0.0
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = HANConv(
            in_channels={"drug": hidden_dim},
            out_channels=hidden_dim,
            metadata=(['drug'], [(f'drug', f'edge_type_{i}', 'drug') for i in range(num_edge_types)]),
            heads=heads,
            dropout=0.0
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.mix_model = MixModel(input_dim=input_dim+hidden_dim*2, output_dim=hidden_dim, p_dropout=p_dropout)
        self.ddi_mlp = GNNDDIMlp(dim_hidden=hidden_dim, dim_ddi_encode=output_dim, p_dropout=p_dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, data, pretrain_start_index, pretrain_end_index, edge_list=None):
        x = data.x
        edge_index_dict = data.edge_index_dict

        x1 = self.conv1({'drug': x}, edge_index_dict)['drug']
        x1 = self.bn1(x1)
        x2 = self.conv2({'drug': x1}, edge_index_dict)['drug']
        x2 = self.bn2(x2)
        x = torch.concat([x, x1, x2], dim=1)
        x = self.mix_model(x)

        start_features = []
        end_features = []

        if edge_list is None:
            start_features = x[data.edge_index[0][::2]][pretrain_start_index:pretrain_end_index]
            end_features = x[data.edge_index[1][::2]][pretrain_start_index:pretrain_end_index]
        else:
            edge_list = edge_list[::2]
            for edge in edge_list:
                start_idx = edge[0]
                end_idx = edge[1]
                start_features.append(x[start_idx])
                end_features.append(x[end_idx])

            start_features = torch.stack(start_features, dim=0)
            end_features = torch.stack(end_features, dim=0)

        x_ddi = torch.cat([start_features, end_features], dim=1)

        x_ddi_new = x_ddi.unsqueeze(1)
        x_ddi_new = self.ddi_atten_layer(query=x_ddi_new, key=x_ddi_new, value=x_ddi_new)[0]
        x_ddi_new = x_ddi_new.squeeze(1)

        x_ddi = torch.cat([x_ddi, x_ddi_new], dim=1)

        x_ddi = self.ddi_mlp(x_ddi)
        return x_ddi, None, None


class PreTrainHAN(nn.Module):
    def __init__(self, dim_drug_features, dim_hidden, dim_drug_embs, dim_ddi_class, dim_ddi_encode, heads, p_dropout):
        super(PreTrainHAN, self).__init__()
        self.ddi_net = HANModel(input_dim=dim_drug_features, hidden_dim=dim_hidden, output_dim=dim_ddi_encode, dim_drug_embs=dim_drug_embs, num_edge_types=dim_ddi_class, p_dropout=p_dropout, heads=heads)
        self.fc1 = nn.Linear(dim_ddi_encode, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, dim_ddi_class)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, data, pretrain_start_index, pretrain_end_index, edge_list):
        x_ddi, _, _ = self.ddi_net(data, pretrain_start_index, pretrain_end_index, edge_list)
        x = self.fc1(x_ddi)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, x_ddi
    