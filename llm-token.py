#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from certa.explain import CertaExplainer
from certa.utils import merge_sources
from certa.models.utils import from_type

dataset = 'abt_buy'
model_type = 'llm'
model = from_type(model_type)

def predict_fn(x):
    return model.predict(x)

datadir = '/path/to/datasets/' + dataset
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')

test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

idx = 4
rand_row = test_df.iloc[idx]
l_id = int(rand_row['ltable_id'])
l_tuple = lsource.iloc[l_id]
r_id = int(rand_row['rtable_id'])
r_tuple = rsource.iloc[r_id]
rand_row.head()

certa_explainer = CertaExplainer(lsource, rsource)

saliency_df, cf_summary, counterfactual_examples, triangles, lattices = certa_explainer.explain(l_tuple, r_tuple, predict_fn, token=True)

print(saliency_df.to_dict())

print(counterfactual_examples)

