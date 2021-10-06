import tensorflow as tf
import pandas as pd
import numpy as np
from alibi.explainers import Counterfactual, CounterfactualProto
from certa.explain import explain
from certa.utils import merge_sources
from models.utils import from_type

import dice_ml


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

idx = 27
rand_row = test_df.iloc[idx]
l_id = int(rand_row['ltable_id'])
l_tuple = lsource.iloc[l_id]
r_id = int(rand_row['rtable_id'])
r_tuple = rsource.iloc[r_id]
rand_row.head()

# d = dice_ml.Data(dataframe=train_df,
#                  continuous_features=[],
#                  outcome_name='label')
#
# m = dice_ml.Model(model=model, backend='TF2')
# exp = dice_ml.Dice(d, m)
#
# query_instance = rand_row.copy().to_dict()
# dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite")


shape = (1,) + ((len(train_df.columns) - 3),)
target_proba = 1.0
tol = 0.01 # want counterfactuals with p(class)>0.99
target_class = 'other' # any class other than 7 will do
max_iter = 1000
lam_init = 1e-1
max_lam_steps = 10
learning_rate_init = 0.1
feature_range = (train_df.min(),train_df.max())
tf.compat.v1.disable_eager_execution()

# d = dice_ml.Data(dataframe=train_df,
#                  continuous_features=[],
#                  outcome_name='label')
# m = dice_ml.Model(model=predict_fn, backend='TF')
# exp = dice_ml.Dice(d, m)

# cf_proto = CounterfactualProto(predict_fn, shape)
cf = Counterfactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                    target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                    max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                    feature_range=feature_range)

