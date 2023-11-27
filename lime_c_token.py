import pandas as pd
from baselines.lime_c import LimeCounterfactual
from certa.models.utils import get_model
from certa.utils import merge_sources, to_token_df
from certa.local_explain import get_original_prediction, get_row
import os

mtype = 'deeper'
dataset = 'beers'
base_datadir = '../cheapER/datasets/'
exp_dir = 'experiments/lime_c'

modeldir = 'models/saved/' + mtype + '/' + dataset
model_name = mtype
datadir = base_datadir + dataset
model = get_model(mtype, modeldir, datadir, dataset)

samples = 10
test = pd.read_csv(datadir + '/test.csv')
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])[:samples]
train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
train_noids = train_df.copy().astype(str)

for idx in range(10):
    rand_row = test_df.iloc[idx]
    l_id = int(rand_row['ltable_id'])
    l_tuple = lsource.iloc[l_id]
    r_id = int(rand_row['rtable_id'])
    r_tuple = rsource.iloc[r_id]

    cf_dir = exp_dir + dataset + '/' + model_name + '/' + str(idx)
    os.makedirs(cf_dir, exist_ok=True)

    label = rand_row["label"]
    row_id = str(l_id) + '-' + str(r_id)
    item = get_row(l_tuple, r_tuple)
    instance = pd.DataFrame(rand_row).transpose().drop(['ltable_id','rtable_id'], axis=1).astype(str)

    def predict_fn_mojito(x):
        return model.predict(x, mojito=True)

    cf_token_explanation = limec_token_explainer = LimeCounterfactual(model, predict_fn_mojito, None, 0.5,
                                                                         train_noids.columns, time_maximum=300,
                                                                         class_names=['nomatch_score', 'match_score'],
                                                                         token=True).explanation(instance)

    print(cf_token_explanation)