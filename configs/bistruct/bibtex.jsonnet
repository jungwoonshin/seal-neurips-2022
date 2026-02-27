// BiStruct config for Bibtex multi-label classification
// Usage:
//   allennlp train configs/bistruct/bibtex.jsonnet \
//     -s run_bibtex_bistruct \
//     --include-package seal

// Hyperparameters
local num_labels = 159;
local input_dim = 1836;
local hidden_dim = 400;
local label_embed_dim = 64;
local lsa_attn_dim = 64;
local lsa_num_heads = 4;
local backward_bottleneck_dim = 128;
local alpha = 0.1;   // alignment loss weight
local beta = 1.0;    // refinement loss weight
local lr = 0.005;
local batch_size = 32;
local num_epochs = 300;
local patience = 20;
local dropout = 0.4;

{
    "dataset_reader": {
        "type": "arff",
        "num_labels": num_labels
    },
    "validation_dataset_reader": {
        "type": "arff",
        "num_labels": num_labels
    },
    "train_data_path": "./data/bibtex_stratified10folds_meka/Bibtex-fold@(1|2|3|4|5|6).arff",
    "validation_data_path": "./data/bibtex_stratified10folds_meka/Bibtex-fold@(7|8).arff",
    "test_data_path": "./data/bibtex_stratified10folds_meka/Bibtex-fold@(9|10).arff",
    "model": {
        "type": "bistruct-multilabel-classification",
        "num_labels": num_labels,
        "feature_network": {
            "input_dim": input_dim,
            "num_layers": 2,
            "hidden_dims": hidden_dim,
            "activations": ["softplus", "softplus"],
            "dropout": [dropout, 0]
        },
        "label_embeddings": {
            "embedding_dim": hidden_dim,
            "vocab_namespace": "labels"
        },
        "label_embed_dim": label_embed_dim,
        "lsa_attn_dim": lsa_attn_dim,
        "lsa_num_heads": lsa_num_heads,
        "lsa_dropout": 0.1,
        "backward_bottleneck_dim": backward_bottleneck_dim,
        "alpha": alpha,
        "beta": beta,
        "initializer": {
            "regexes": [
                [".*_linear_layers.*weight", {"type": "kaiming_uniform", "nonlinearity": "relu"}],
                [".*linear_layers.*bias", {"type": "zero"}]
            ]
        }
    },
    "data_loader": {
        "batch_size": batch_size,
        "shuffle": true
    },
    "trainer": {
        "type": "gradient_descent",
        "optimizer": {
            "type": "adamw",
            "lr": lr,
            "weight_decay": 1e-5
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 5,
            "verbose": true
        },
        "num_epochs": num_epochs,
        "patience": patience,
        "validation_metric": "+fixed_f1",
        "grad_norm": 10.0,
        "cuda_device": -1,
        "callbacks": [
            "track_epoch_callback"
        ],
        "checkpointer": {
            "keep_most_recent_by_count": 1
        }
    },
    "evaluate_on_test": true,
    "type": "train_test_log_to_wandb"
}
