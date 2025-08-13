# adaptive lr
gnn_fine_tuning_factor = 0.1
gnn_fine_tuning_patience = 5
gnn_fine_tuning_threshold = 0.1

# gnn_pre_train
gnn_pretrain_lr = 1e-3
gnn_pretrain_epochs = 2000
gnn_pretrain_p_dropout = 0
gnn_pretrain_l2_weight = 0
gnn_pretrain_batch_size = 2 ** 17

# gnn_fine_tuning
gnn_finetuning_batch_size = 512
gnn_fine_tuning_epochs = 50
gnn_finetuning_l2_weight = 1e-3
heads = 2

# dimensions of embeddings
gnn_dim_drug_embs = 256
gnn_dim_cell_embs = 256
gnn_dim_ddi_encode = 64
gnn_dim_hidden = 40