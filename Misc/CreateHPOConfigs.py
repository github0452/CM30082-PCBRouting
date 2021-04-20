import random
import json
from pathlib import Path

general_config = {
    'save_csv': 'False',
    'save_tensor': 'False',
    'n_batch': '10',
    'n_batch_train_size': '512',
    'n_batch_test_size': '5120',
    'baseline_type': 'Critic'
}
con_pntr_config = {
    'processing': [0, 1, 2],
    'hidden': [128, 256, 512],
    'glimpse': [0, 1, 2],
    'max_grad': [0.5, 1, 2],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'learning_rate_gamma': [1.0, 0.98, 0.96]
}
con_trans_config = {
    'n_layers': [1, 2, 4],
    'n_head': [1, 4, 8],
    'dim_model': [64, 128, 256],
    'dim_hidden': [128, 256, 512],
    'dim_v': [32, 64, 128],
    'max_grad': [0.5, 1, 2],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'learning_rate_gamma': [1.0, 0.98, 0.96]
}
improve_config = {
    'n_layers': [1, 2, 4],
    'n_head': [1, 4, 8],
    'dim_model': [128],
    'dim_hidden': [64],
    'dim_v': [32, 64, 128],
    'max_grad': [0.5, 1, 2],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'learning_rate_gamma': [1.0, 0.98, 0.96],
    't': '1'
}
models = [
    ('PointerNetwork', 'Construction', con_pntr_config),
    ('Transformer', 'Construction', con_trans_config),
    ('TSP_improve', 'Improvement', improve_config)]

DATA = "runs"
OPTIONS = 32

for model_name, env_name, model_config in models:
    folder = "{0}/{1}".format(DATA, model_name)
    Path(folder).mkdir(parents=True, exist_ok=True)
    for i in range(OPTIONS):
        config = {
            'model': model_name,
            'environment': env_name,
            'data_path': "{0}/config{1}".format(folder, i)
        }
        config.update(general_config)
        for parameter, options in model_config.items():
            config[parameter] = random.choice(options)
        # randomly select parameters
        file_name = "{0}/configFiles/{1}-{2}".format(DATA, model_name, i)
        print(file_name)
        with open(file_name, 'w') as outfile:
            json.dump(config, outfile, indent=4)
