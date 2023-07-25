#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from certa.explain import CertaExplainer
from certa.utils import merge_sources
from certa.models.utils import from_type

dataset = 'beers'
model_type = 'chatgpt'
model = from_type(model_type)
n_samples = 5

do_attr = True
do_token = True

def predict_fn(x):
    return model.predict(x)

datadir = '/home/tteofili/dev/cheapER/datasets/' + dataset
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')

test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

certa_explainer = CertaExplainer(lsource, rsource)
attr_saliencies = []
token_saliencies = []
attr_cfs = []
token_cfs = []
attr_triangles_count = 0
token_triangles_count = 0
for idx in range(n_samples):
    rand_row = test_df.iloc[idx]
    l_id = int(rand_row['ltable_id'])
    l_tuple = lsource.iloc[l_id]
    r_id = int(rand_row['rtable_id'])
    r_tuple = rsource.iloc[r_id]
    if do_attr:
        attr_saliency_df, attr_cf_summary, attr_counterfactual_examples, attr_triangles, attr_lattices = certa_explainer.explain(l_tuple, r_tuple, predict_fn, token=False, num_triangles=10)
        attr_saliencies.append(attr_saliency_df)
        attr_cfs.append(attr_counterfactual_examples)
        attr_triangles_count += len(attr_triangles)
        print(f'avg attr triangles: {attr_triangles_count / (idx + 1)}')
        pd.concat(attr_saliencies).to_csv('certa_attr_saliencies.csv')
        pd.concat(attr_cfs).to_csv('certa_attr_cfs.csv')
    if do_token:
        token_saliency_df, token_cf_summary, token_counterfactual_examples, token_triangles, token_lattices = certa_explainer.explain(l_tuple, r_tuple, predict_fn, token=True, num_triangles=5)
        token_saliencies.append(token_saliency_df)
        token_cfs.append(token_counterfactual_examples)
        token_triangles_count += len(token_triangles)
        print(f'avg token triangles: {token_triangles_count / (idx + 1)}')
        pd.concat(token_saliencies).to_csv('certa_token_saliencies.csv')
        pd.concat(token_cfs).to_csv('certa_token_cfs.csv')

