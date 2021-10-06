import traceback

import sklearn.pipeline
import tensorflow as tf
import pandas as pd
import numpy as np
from alibi.explainers import Counterfactual, CounterfactualProto
from sklearn.base import BaseEstimator

from baselines.shap_c import ShapCounterfactual
from certa.explain import explain
from certa.utils import merge_sources
from models.utils import from_type

import dice_ml

dice = True
proto = False
simple = False
shap_c = False

dataset = 'beers'
model_type = 'dm'
model = from_type(model_type)
model.load('models/' + model_type + '/' + dataset)

def predict_fn(x):
    return model.predict_proba(x)

datadir = 'datasets/' + dataset
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')

test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, [], ['label'])
train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, [], ['label'])

test_df['outcome'] = np.argmax(model.predict_proba(test_df), axis=1)

tf.compat.v1.disable_eager_execution()
for idx in range(10):
    rand_row = test_df.iloc[idx]
    l_id = int(rand_row['ltable_id'])
    l_tuple = lsource.iloc[l_id]
    r_id = int(rand_row['rtable_id'])
    r_tuple = rsource.iloc[r_id]
    rand_row.head()

    classifier_fn = lambda x: model.predict_proba(x, given_columns=train_df.columns)[1, :]
    if shap_c:
        print('shap-c')
        try:
            shapc_explainer = ShapCounterfactual(classifier_fn, 0.5,
                     train_df.columns)
            sc_exp = shapc_explainer.explanation(pd.DataFrame(rand_row).transpose())
            print(f'{idx}- shap-c:{sc_exp}')
        except:
            pass

    if dice:
        print('dice')
        try:
            d = dice_ml.Data(dataframe=test_df,
                          continuous_features=[],
                          outcome_name='outcome')

            m = dice_ml.Model(model=model, backend='sklearn')
            exp = dice_ml.Dice(d, m, method='random')
            query_instance = rand_row.copy().to_dict()
            dice_exp = exp.generate_counterfactuals(pd.DataFrame(rand_row).transpose().drop(['outcome'], axis=1),
                                                    total_CFs=5, desired_class="opposite")
            dice_exp_df = dice_exp.visualize_as_dataframe()
            print(f'{idx}:{dice_exp_df}')
        except:
            pass


    shape = (1,) + ((len(train_df.columns) - 2),)

    instance = pd.DataFrame(rand_row).transpose().drop(['ltable_id', 'rtable_id'], axis=1).values
    if proto:
        print('proto')
        try:
            cf_proto = CounterfactualProto(predict_fn, shape, feature_range=(str(train_df.min(axis=0)), str(train_df.max(axis=0))))
            cf_proto.fit(train_df.drop(['ltable_id', 'rtable_id'], axis=1).values)
            proto_ex = cf_proto.explain(instance)
            print(f'{proto_ex}')
        except:
            print(traceback.format_exc())
            print(f'skipped item {str(idx)}')
            pass

    if simple:
        print('simple')
        try:
            cf = Counterfactual(predict_fn, shape=shape, feature_range=(str(train_df.min(axis=0)), str(train_df.max(axis=0))))
            simple_ex = cf.explain(instance)
            print(f'{simple_ex}')
        except:
            print(traceback.format_exc())
            print(f'skipped item {str(idx)}')
            pass

