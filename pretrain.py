from parameters.parameters_common import *
from train_utils import *
from utils import *


def gnn_pre_train(continue_pretrain, pretrain_pth):
    device = get_device()

    logging.info('开始gnn预训练！')
    model = PreTrainHAN(dim_drug_features=dim_drug_features, dim_drug_embs=gnn_dim_drug_embs,
                            dim_hidden=gnn_dim_hidden,
                            dim_ddi_class=dim_ddi_class, p_dropout=gnn_pretrain_p_dropout,
                            dim_ddi_encode=gnn_dim_ddi_encode, heads=heads).to(device)


    if continue_pretrain:
        model.load_state_dict(torch.load(pretrain_pth))

    _, y_train = load_pre_train_dataset()

    criterion = FocalLoss(alpha=None, gamma=2, size_average=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=gnn_pretrain_lr, weight_decay=gnn_pretrain_l2_weight)

    ddi_graph, _, _, _ = load_ddi_graph_data()

    ddi_graph = get_heterogeneous_data(ddi_graph)

    print('开始训练...')
    min_train_loss = 1e9
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    writer = SummaryWriter(comment='pretrain_gnn')
    for epoch in range(1, gnn_pretrain_epochs + 1):
        ddi_loader = batch_edges(ddi_graph, batch_size=gnn_pretrain_batch_size)

        epoch_total_loss = 0.0
        epoch_total_acc = 0.0
        pbar_ddi_loader = tqdm(ddi_loader, desc=f'第{epoch}轮GNN-pretrain训练')
        # flag = False
        for index, line in enumerate(pbar_ddi_loader):
            batch_edge_list = line[0]
            batch_edge_type = line[1]
            labels = torch.Tensor(batch_edge_type).to(torch.int).to(get_device())

            labels = labels[::2]

            batch_size = len(labels)
            batch_start_index = batch_size * index
            batch_end_index = batch_size * (index + 1)

            optimizer.zero_grad()
            outputs, x_ddi = model(data=ddi_graph, pretrain_start_index=batch_start_index, pretrain_end_index=batch_end_index,
                            edge_list=None)

            model.train()
            epoch_train_loss = criterion(outputs, labels.long())

            epoch_train_loss.backward()
            optimizer.step()

            epoch_train_acc = calculate_acc(outputs, labels)

            epoch_total_loss += epoch_train_loss
            epoch_total_acc += epoch_train_acc

            pbar_ddi_loader.set_postfix(loss=epoch_train_loss.item())

        epoch_train_loss = epoch_total_loss / len(ddi_loader)
        epoch_train_acc = epoch_total_acc / len(ddi_loader)

        train_loss_list.append(epoch_train_loss.item())
        train_acc_list.append(epoch_train_acc)

        logging.info(f'第{epoch}轮pretrain训练, train_loss={epoch_train_loss}, train_acc={epoch_train_acc}')

        if epoch_train_loss < min_train_loss:
            min_train_loss = epoch_train_loss
            min_train_acc = epoch_train_acc
            torch.save(model.state_dict(), pretrain_pth)

            writer.add_scalars(f'gnn_pretrain_acc', {'train_acc': epoch_train_acc}, epoch)
            writer.add_scalars(f'gnn_pretrain_loss', {'train_loss': epoch_train_loss}, epoch)

    writer.close()
    logging.info(f'预训练的最好train_acc为{min_train_acc}', 'gnn')

    logging.info(f'训练完成！ 模型已保存！')


if __name__ == '__main__': 
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(f'./logs/gnn_pretrain.log', mode='a', encoding='utf-8')
                    ])

    reset_seed(42)
    gnn_pre_train(continue_pretrain=False, pretrain_pth='./checkpoints/pretrained_han.pth')
    