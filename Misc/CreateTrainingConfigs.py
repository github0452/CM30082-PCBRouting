import random
import json
from pathlib import Path

general_config = {
    'save_csv': 'True',
    'save_tensor': 'True',
    'n_batch': '10',
    'n_batch_train_size': '512',
    'n_batch_test_size': '5120',
    'baseline_type': 'Critic'
}
con_pntr_config = {
    'processing': 0,
    'hidden': 512,
    'glimpse': 0,
    'max_grad': 2,
    'learning_rate': 1e-4,
    'learning_rate_gamma': 0.98
}
con_trans_config = {
    'n_layers': 2,
    'n_head': 4,
    'dim_model': 128,
    'dim_hidden': 256,
    'dim_v': 64,
    'max_grad': 1,
    'learning_rate': 1e-4,
    'learning_rate_gamma': 1.0
}
improve_config = {
    'n_layers': 4,
    'n_head': 1,
    'dim_model': 128,
    'dim_hidden': 64,
    'dim_v': 32,
    'max_grad': 1,
    'learning_rate': 1e-3,
    'learning_rate_gamma': 1.0,
    't': '1'
}
models = [
    ('PointerNetwork', 'Construction', con_pntr_config),
    ('Transformer', 'Construction', con_trans_config),
    ('TSP_improve', 'Improvement', improve_config)]

RUNS = 9
for model_name, env_name, model_config in models:
    model_folder = "finalTraining"
    config_folder = "{0}/config".format(model_folder)
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    Path(config_folder).mkdir(parents=True, exist_ok=True)
    for i in range(RUNS):
        config = {
            'model': model_name,
            'environment': env_name,
            'data_path': "{0}/{1}_run{2}".format(model_folder, model_name, i)
        }
        config.update(general_config)
        config.update(model_config)
        config_file = "{0}/{1}-run{2}".format(config_folder, model_name, i)
        with open(config_file, 'w') as outfile:
            json.dump(config, outfile, indent=4)
