import logging

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from data_loader import *
from model import *


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1,input.size(2))
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def transfer_pretrain_han(fine_tuning_model: HGTSynergy, dim_drug_features, dim_drug_embs, dim_hidden, dim_ddi_encode, dim_ddi_class, p_dropout, heads, pretrain_pth):
    print('开始加载HAN预训练模型...')

    device = get_device()

    # 初始化模型但不指定设备
    pre_train_model = PreTrainHAN(
        dim_drug_features=dim_drug_features,
        dim_drug_embs=dim_drug_embs,
        dim_hidden=dim_hidden,
        dim_ddi_encode=dim_ddi_encode,
        dim_ddi_class=dim_ddi_class,
        p_dropout=p_dropout,
        heads=heads,
    )

    pre_train_model.load_state_dict(torch.load(pretrain_pth, map_location='cpu'))

    pre_train_model.to(device)

    fine_tuning_model.ddi_net.load_state_dict(pre_train_model.ddi_net.state_dict())

    return fine_tuning_model



def train_model(model, use_pretrain_model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_pth, name, model_type, ddi_graph=None):
    logger = logging.getLogger(name)
    logger.info(f'开始{name}训练...')
    min_train_avg_loss = 1e9
    min_val_avg_loss = 1e9
    writer = SummaryWriter(comment=f'{name}训练, use_pretrain_model={use_pretrain_model}')
    for epoch in range(1, epochs + 1):
        epoch_total_loss = 0

        model.train()
        pbar_train_loader = tqdm(train_loader, desc=f'第{epoch}轮{name}训练')
        for line in pbar_train_loader:
            optimizer.zero_grad()

            drugA_features_tensor, drugB_features_tensor, cell_features_tensor, labels_reg, labels_cls = line[0]
            ddi_index_tensor = line[1]
            outputs = model(drugA_features_tensor, drugB_features_tensor, cell_features_tensor, ddi_graph, ddi_index_tensor)

            loss = criterion(outputs, labels_reg)

            epoch_total_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar_train_loader.set_postfix(loss=loss.item())

        epoch_train_avg_loss = epoch_total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loader_iter = iter(val_loader)
            line = next(val_loader_iter)

            drugA_features_tensor, drugB_features_tensor, cell_features_tensor, labels_reg, labels_cls = line[0]
            ddi_index_tensor = line[1]
            outputs = model(drugA_features_tensor, drugB_features_tensor, cell_features_tensor, ddi_graph, ddi_index_tensor)

            val_loss = criterion(outputs, labels_reg)


        epoch_val_avg_loss = val_loss.item() / len(val_loader)
        scheduler.step(epoch_val_avg_loss)


        min_train_avg_loss = min(min_train_avg_loss, epoch_train_avg_loss)
        if epoch_val_avg_loss < min_val_avg_loss:
            min_val_avg_loss = epoch_val_avg_loss
            torch.save(model.state_dict(), model_pth)
        

        logger.info(f"第{epoch}轮训练的结果: train_loss: {epoch_train_avg_loss}, val_loss: {epoch_val_avg_loss}")
        writer.add_scalars(f'{name}训练过程Loss', {'train_loss': epoch_train_avg_loss, 'val_loss': epoch_val_avg_loss}, epoch)

        tqdm.write('第{}轮{}训练，Train_Loss: {}, Val_Loss: {}'.format(epoch, name, epoch_train_avg_loss, epoch_val_avg_loss))
    
    logger.info(f"{name}训练完成, 最小的train_loss为{min_train_avg_loss}, 最小的val_loss为{min_val_avg_loss}")
    writer.close()
    return min_train_avg_loss, min_val_avg_loss


def test_model(model, test_loader, criterion, optimizer, name, df_test_data, ddi_graph=None, model_pth=None):
    logging.info(f"开始测试{name}模型")
    model.load_state_dict(torch.load(model_pth))
    optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        test_loader_iter = iter(test_loader)
        line = next(test_loader_iter)

        drugA_features_tensor, drugB_features_tensor, cell_features_tensor, labels_reg, labels_cls = line[0]
        ddi_index_tensor = line[1]
        outputs = model(drugA_features_tensor, drugB_features_tensor, cell_features_tensor, ddi_graph, ddi_index_tensor)

        test_loss = criterion(outputs, labels_reg).item()

        predictions = outputs.cpu().detach().numpy()
        targets = labels_reg.cpu().detach().numpy()

        test_rmse = calculate_rmse(predictions, targets)
        test_pcc = calculate_pcc(predictions, targets)

        logging.info(f'{name}测试: test_mse: {test_loss}, test_rmse: {test_rmse}, test_pcc: {test_pcc}')

        with open("./data/cell2tissue.json", 'r') as json_file:
            cell2tissue_dir = json.load(json_file)

        predictions = predictions[::2]

        df_test_data['predicted_synergy'] = predictions

        # 记录细胞系所属的组织
        df_test_data['tissue'] = '0'
        for index, row in df_test_data.iterrows():
            df_test_data.loc[index, 'tissue'] = cell2tissue_dir[row['cell_line']]

        df_test_data.to_csv(f'./output/{name}_predictions.csv')

        return test_loss, test_rmse, test_pcc
