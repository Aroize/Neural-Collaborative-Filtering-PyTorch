from hparams.utils import Hparam

"""
As NeuMF config consists of two configs:
* GMF config
* MLP config
It is really easy to fail and forgot to change one of parameters
in config, while tuning hyperparameters
As each model contains both unique values and common, we can create
three configs, using single lines of data in config (with no repetition)
"""
def generate_neu_mf_config(config_path):
    config = Hparam(config_path)
    gmf_config = config.gmf
    mlp_config = config.mlp
    neu_mf_config = config.neu_mf
    
    for k,v in mlp_config.items():
        setattr(neu_mf_config, k, v)
    for k,v in gmf_config.items():
        setattr(neu_mf_config, k, v)
    
    configs = [gmf_config, mlp_config, neu_mf_config]
    for conf in configs:
        conf.user_count = config.user_count
        conf.item_count = config.item_count
    return configs


if __name__ == '__main__':
    import sys
    config_path = sys.argv[1]
    gmf, mlp, neu = generate_neu_mf_config(config_path)
    print('GMF config ', gmf)
    print('MLP config ', mlp)
    print('NeuMF config', neu)