#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from certa.explain import CertaExplainer
from certa.utils import merge_sources
from certa.models.utils import from_type, get_model

dataset = 'abt_buy'
mtype = 'deeper'
base_datadir = '../cheapER/datasets/'
modeldir = 'models/saved/' + mtype + '/' + dataset
datadir = base_datadir + dataset
model = get_model(mtype, modeldir, datadir, dataset)

def predict_fn(x):
    return model.predict(x)

lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
train = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')
test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])
certa_explainer = CertaExplainer(lsource, rsource)

idxs = [3, 379, 384]

for idx in idxs:
    rand_row = test_df.iloc[idx]
    l_id = int(rand_row['ltable_id'])
    l_tuple = lsource.iloc[l_id]
    r_id = int(rand_row['rtable_id'])
    r_tuple = rsource.iloc[r_id]
    print(f'label:{rand_row["label"]}, prediction:{predict_fn(pd.DataFrame(rand_row).T)["match_score"].values[0]}')
    saliency_df, cf_summary, counterfactual_examples, triangles, lattices = certa_explainer.explain(l_tuple, r_tuple, predict_fn, token=True, num_triangles=10)
    saliency_df.to_csv(str(idx)+'_saliency.csv')
    counterfactual_examples.to_csv(str(idx)+'_cf.csv')
