from hparams.utils import Hparam
import math

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


"""
NDCG@K - Normalized Discounted Cumulative Gain at K

$ DCG@K = \sum_{i = 1}^{N} \frac{rel_i}{log_2(i + 1)} $
$ NDCG@K = \frac{DCG@K}{IDCG@K} $

:param: predictions - list of tuples (label, score)
:param: test_items - set of labels, representing real items in test sample
:param: topK - take this count of predictions and calculate metrics
"""
def ndcg(test_items, predictions, topK):
    assert len(test_items) > 0, "Set of items must be not empty"
    assert len(predictions) >= topK, "Count of predictions must be greater than K value"
    topK_predictions = sorted(predictions, key=lambda x: x[1])[:topK]
    dcg_score = 0.0
    for index, data in enumerate(topK_predictions, 1):
        label, score = data
        if label in test_items:
            dcg_score += (1.0 / math.log(index + 1, 2))
    ideal_count = min(topK, len(test_items))
    ideal_score = 0.0
    for index in range(ideal_count):
        ideal_score += (1.0 / math.log(index + 2, 2))
    return float(dcg_score) / ideal_score


if __name__ == '__main__':
    import sys
    config_path = sys.argv[1]
    gmf, mlp, neu = generate_neu_mf_config(config_path)
    print('GMF config ', gmf)
    print('MLP config ', mlp)
    print('NeuMF config', neu)
    
    