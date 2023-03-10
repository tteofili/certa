import pandas as pd
from baselines.minun_explainer import MinunExplainer, formulate_instance
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

minun_explainer = MinunExplainer(model)

for idx in range(10):
    cf_dir = exp_dir + dataset + '/' + model_name + '/' + str(idx)
    os.makedirs(cf_dir, exist_ok=True)

    instance = formulate_instance(lsource, rsource, test.iloc[idx])

    cf_token_explanation = minun_explainer.explain(instance)
    print(cf_token_explanation)
