from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from toolz.dicttoolz import valmap, keyfilter

import numpy as np
from functools import partial

from models import GPN_Encoder
import datasets
import main

config = {
    'dataset_name': 'Cora',
    'sampler': 'model',
    'budget': 98,
    'seed': 3133742069,
    'runs': 10,
    'average_repeats': True,
    'hyperparameters':{
        'embedding_dim': tune.choice(range(12,24,2)),
        'hidden_dim_size': tune.choice(range(96,196,32)),
        'dropout': tune.uniform(0.2,0.8),
        'distance_loss_weight': tune.loguniform(0.25,4),
        'train_epochs': tune.choice(range(3,10)),
        'learning_rate': tune.loguniform(1e-3,1e-2)}}

config1 = {
    'dataset_name': 'Cora',
    'sampler': 'model',
    'budget': 98,
    'seed': 3133742069,
    'runs': 3,
    'average_repeats': True,
    'hyperparameters':{
        'embedding_dim': 16,
        'hidden_dim_size': 96,
        'dropout': 0.5,
        'distance_loss_weight': 1,
        'train_epochs': 4,
        'learning_rate': 0.001,}}

scheduler = ASHAScheduler(
    max_t=1,
    grace_period=1,
    reduction_factor=2)

def train(config):
    dataset = datasets.get_dataset(config["dataset_name"])
    hyperparams = config['hyperparameters']
    gpn_model = partial(GPN_Encoder,
                        num_node_features=dataset.num_node_features,
                        num_classes=dataset.num_classes,
                        pagerank_scores=dataset.pagerank,
                        **keyfilter(lambda x: x in ['hidden_dim_size',
                                                    'embedding_dim',
                                                    'dropout',
                                                    'distance_loss_weight'],
                                    hyperparams))
    results = main.run(model=gpn_model,
                 dataset=dataset,
                 **keyfilter(lambda x: x in ['sampler',
                                             'runs',
                                             'budget',
                                             'seed'],
                             config),
                 **keyfilter(lambda x: x in ['train_epochs',
                                             'learning_rate'],
                             hyperparams))
    f1s = list(map(lambda x: x['test']['macro-f1'][-1], results))
    return {"macro-f1": np.mean(f1s)}

tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train),
            resources={"cpu": 1, "gpu": 0}
        ),
        tune_config=tune.TuneConfig(
            metric="macro-f1",
            mode="max",
            scheduler=scheduler,
            num_samples=1000,
        ),
        param_space=config,
    )
result = tuner.fit()
