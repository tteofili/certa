import sklearn.pipeline
import tensorflow as tf
import pandas as pd
import numpy as np
from alibi.explainers import Counterfactual, CounterfactualProto
from sklearn.base import BaseEstimator

from certa.explain import explain
from certa.utils import merge_sources
from models.utils import from_type

import dice_ml

dice = True
proto = False
simple = False

dataset = 'beers'
model_type = 'dm'
model = from_type(model_type)
model.load('models/' + model_type + '/' + dataset)

def predict_fn(x):
    return model.predict(x, mojito=True, expand_dim=True)

datadir = 'datasets/' + dataset
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')

test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])
train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

#tf.compat.v1.disable_eager_execution()
for idx in range(10):
    rand_row = test_df.iloc[idx]
    l_id = int(rand_row['ltable_id'])
    l_tuple = lsource.iloc[l_id]
    r_id = int(rand_row['rtable_id'])
    r_tuple = rsource.iloc[r_id]
    rand_row.head()

    if dice:

        d = dice_ml.Data(dataframe=pd.concat([train_df, test_df]),
                          continuous_features=[],
                          outcome_name='label')

        m = dice_ml.Model(model=model, backend='sklearn')
        exp = dice_ml.Dice(d, m, method='random')
        query_instance = rand_row.copy().to_dict()
        dice_exp = exp.generate_counterfactuals(pd.DataFrame(rand_row).transpose().drop(['label'], axis=1),
                                                total_CFs=5, desired_class="opposite")
        dice_exp_df = dice_exp.visualize_as_dataframe()
        print(f'{idx}:{dice_exp_df}')


    shape = (1,) + ((len(train_df.columns)),)
    target_proba = 1.0
    tol = 0.01 # want counterfactuals with p(class)>0.99
    target_class = 'other' # any class other than 7 will do
    max_iter = 1000
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (0,1)


    if proto:
        instance = pd.DataFrame(rand_row).transpose().drop(['label', 'ltable_id', 'rtable_id'], axis=1)
        cf_proto = CounterfactualProto(predict_fn, shape, feature_range=(train_df.min(axis=0), train_df.max(axis=0)))
        cf_proto.fit(train_df.values)
        cf_proto.explain(instance)

    if simple:
        cf = Counterfactual(predict_fn, shape=shape, feature_range=(train_df.min(axis=0), train_df.max(axis=0)))

