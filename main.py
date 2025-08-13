import argparse
import itertools

from torch.optim.lr_scheduler import ReduceLROnPlateau

from train_utils import *
from utils import *
from parameters.parameters_common import *


def nest_cv(lr_list, dropout_list, hidden_size_list, model_pth, use_pretrain_model=True,
            pretrain_pth=None, no=None, start_test_fold=0):

    device = get_device()
    logging.info(f'训练配置：use_pretrain_model={use_pretrain_model}')
    print('开始加载Graph数据...')
    ddi_graph, drugbank_id_to_index, edge_index_dict, edge_type_dict = load_ddi_graph_data()

    ddi_graph = get_heterogeneous_data(ddi_graph)

    test_mse_list = []
    test_rmse_list = []
    test_pcc_list = []

    for test_fold in range(5):
        if test_fold < start_test_fold:
            continue

        min_mean_val_mse = 1e9
        best_lr = -1
        best_dropout = -1
        best_hidden_size = -1

        for LR, DROPOUT, HIDDEN_SIZE in itertools.product(lr_list, dropout_list, hidden_size_list):
            # 如果只设置了一组参数，则不需要调参
            if len(lr_list) == 1 and len(dropout_list) == 1 and len(hidden_size_list) == 1:
                best_lr = LR
                best_dropout = DROPOUT
                best_hidden_size = HIDDEN_SIZE
                break

            val_mse_list = []
            for val_fold in range(5):
                if test_fold == val_fold:
                    continue

                logging.info("================================================================")
                logging.info(
                    f"正在进行嵌套交叉验证, test_fold={test_fold}, lr={LR,}, dropout={DROPOUT}, hidden_size={HIDDEN_SIZE}, val_fold={val_fold}")
                x_train, y_train, x_val, y_val, x_test, y_test = load_nest_cv_fine_tuning_dataset(test_fold=test_fold,
                                                                                                  val_fold=val_fold)
                train_loader, val_loader, _test_loader = get_fine_tuning_loaders(x_train, y_train, x_val, y_val, x_test,
                                                                                 y_test,
                                                                                 drugbank_id_to_index=drugbank_id_to_index)

                reset_seed(42)
                model = HGTSynergy(dim_drug_features=dim_drug_features, dim_cell_features=dim_cell_features,
                                   dim_drug_embs=gnn_dim_drug_embs, dim_cell_embs=gnn_dim_cell_embs,
                                   dim_ddi_encode=gnn_dim_ddi_encode, dim_ddi_class=dim_ddi_class,
                                   dim_hidden=gnn_dim_hidden, hidden_size=HIDDEN_SIZE,
                                   p_dropout=DROPOUT, heads=heads).to(device)

                if use_pretrain_model:
                    model = transfer_pretrain_han(model, dim_drug_features, gnn_dim_drug_embs, gnn_dim_hidden,
                                                  gnn_dim_ddi_encode, dim_ddi_class, gnn_pretrain_p_dropout, heads,
                                                  pretrain_pth)

                criterion = nn.MSELoss()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=best_lr,
                    weight_decay=gnn_finetuning_l2_weight
                )

                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gnn_fine_tuning_factor,
                                              patience=gnn_fine_tuning_patience, threshold=gnn_fine_tuning_threshold,
                                              verbose=True)

                min_train_avg_loss, min_val_avg_loss = train_model(model=model,
                                                                   use_pretrain_model=use_pretrain_model,
                                                                   train_loader=train_loader,
                                                                   val_loader=val_loader,
                                                                   criterion=criterion,
                                                                   optimizer=optimizer,
                                                                   scheduler=scheduler,
                                                                   epochs=gnn_fine_tuning_epochs,
                                                                   model_pth=model_pth,
                                                                   name=f"嵌套交叉验证模型",
                                                                   model_type='gnn',
                                                                   ddi_graph=ddi_graph)

                val_mse_list.append(min_val_avg_loss)

                mean_val_mse = np.mean(val_mse_list)
                if mean_val_mse < min_mean_val_mse:
                    min_mean_val_mse = mean_val_mse
                    best_lr = LR
                    best_dropout = DROPOUT
                    best_hidden_size = HIDDEN_SIZE

        logging.info(
            f"嵌套交叉验证test_fold={test_fold}中, best_lr={best_lr}, best_dropout={best_dropout}, best_hidden_size={best_hidden_size}")

        reset_seed(42)

        model = HGTSynergy(dim_drug_features=dim_drug_features, dim_cell_features=dim_cell_features,
                           dim_drug_embs=gnn_dim_drug_embs, dim_cell_embs=gnn_dim_cell_embs,
                           dim_ddi_encode=gnn_dim_ddi_encode, dim_ddi_class=dim_ddi_class, dim_hidden=gnn_dim_hidden,
                           p_dropout=best_dropout, heads=heads, hidden_size=best_hidden_size).to(device)

        if use_pretrain_model:
            model = transfer_pretrain_han(model, dim_drug_features, gnn_dim_drug_embs, gnn_dim_hidden,
                                          gnn_dim_ddi_encode, dim_ddi_class, gnn_pretrain_p_dropout, heads,
                                          pretrain_pth)

        criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_lr,
            weight_decay=gnn_finetuning_l2_weight
        )

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gnn_fine_tuning_factor,
                                      patience=gnn_fine_tuning_patience, threshold=gnn_fine_tuning_threshold,
                                      verbose=True)

        x_outer_train_list, y_outer_train_list, x_test, y_test = load_nest_test_dataset(test_fold=test_fold)
        x_outer_train, x_outer_val, y_outer_train, y_outer_val = split_into_train_val(x_outer_train_list,
                                                                                      y_outer_train_list,
                                                                                      split_rate=0.9)
        outer_train_loader, outer_val_loader, outer_test_loader = get_fine_tuning_loaders(x_outer_train, y_outer_train,
                                                                                          x_outer_val, y_outer_val,
                                                                                          x_test, y_test,
                                                                                          drugbank_id_to_index=drugbank_id_to_index)

        min_train_avg_loss, min_val_avg_loss = train_model(model=model,
                                                           use_pretrain_model=use_pretrain_model,
                                                           train_loader=outer_train_loader,
                                                           val_loader=outer_val_loader,
                                                           criterion=criterion,
                                                           optimizer=optimizer,
                                                           scheduler=scheduler,
                                                           epochs=gnn_fine_tuning_epochs,
                                                           model_pth=model_pth,
                                                           name=f"嵌套交叉验证模型",
                                                           model_type='gnn',
                                                           ddi_graph=ddi_graph)

        # 测试集上测试
        synergy_file = f'./data/cv_labels_all.csv'
        all_synergy_data = get_synergy_data(synergy_file=synergy_file)
        df_test_data = all_synergy_data[all_synergy_data['fold'] == test_fold]

        test_mse, test_rmse, test_pcc = test_model(model, outer_test_loader, criterion, optimizer,
                                                   name=f'nest_pretrain={use_pretrain_model}_no={no}_testFold={test_fold}',
                                                   ddi_graph=ddi_graph, model_pth=model_pth, df_test_data=df_test_data)

        test_mse_list.append(test_mse)
        test_rmse_list.append(test_rmse)
        test_pcc_list.append(test_pcc)
        logging.info("*******************************************************")
        logging.info(
            f'test_fold={test_fold}的嵌套训练完成！use_pretrain_model=={use_pretrain_model}。最好的参数组合: best_lr={best_lr}, best_dropout={best_dropout}, best_hidden_size={best_hidden_size}。在最终训练以后，test_mse为{test_mse}, ' +
            f'test_rmse为{test_rmse}，test_pcc为{test_pcc}')

    mse_mean, mse_std = calculate_mean_and_std(test_mse_list)
    rmse_mean, rmse_std = calculate_mean_and_std(test_rmse_list)
    pcc_mean, pcc_std = calculate_mean_and_std(test_pcc_list)
    logging.info("******************************************************")
    logging.info(
        f"嵌套交叉验证全部完成！use_pretrain_model={use_pretrain_model}。mse={mse_mean}±{mse_std}, rmse={rmse_mean}±{rmse_std}, pcc={pcc_mean}±{pcc_std}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(f'./logs/HGTSynergy.log', mode='a', encoding='utf-8')
                    ])


    pretrain_pth = f'./checkpoints/pretrained_han.pth'

    nest_cv(lr_list=[0.45e-3], dropout_list=[0.20], hidden_size_list=[2048, 4096, 8192], use_pretrain_model=True,
        model_pth=f'./checkpoints/HGTSynergy.pth',
        pretrain_pth=pretrain_pth,
        no='1',
        start_test_fold=0)

