import logging
import os
import time
import traceback

import numpy as np
import pandas as pd
from certa.explain import explain
from certa.local_explain import get_original_prediction, get_row
from certa.utils import merge_sources

from baselines.landmark import Landmark
from baselines.mojito import Mojito
import shap

from models.ermodel import ERModel
from models.utils import from_type

root_datadir = 'datasets/'
experiments_dir = 'cf/'


def evaluate(model: ERModel, samples: int = 50, filtered_datasets: list = [], exp_dir: str = experiments_dir,
             fast: bool = False, max_predict: int = -1):
    if not exp_dir.endswith('/'):
        exp_dir = exp_dir + '/'

    for subdir, dirs, files in os.walk(root_datadir):
        for dir in dirs:
            if dir not in filtered_datasets:
                continue
            for robust in [False]:
                os.makedirs(exp_dir + dir, exist_ok=True)
                model_name = model.name
                if robust:
                    model_name = model_name + '_robust'
                os.makedirs(exp_dir + dir + '/' + model_name, exist_ok=True)
                if dir == 'temporary':
                    continue
                print(f'working on {dir}')
                datadir = os.path.join(root_datadir, dir)
                logging.info(f'reading data from {datadir}')

                lsource = pd.read_csv(datadir + '/tableA.csv')
                rsource = pd.read_csv(datadir + '/tableB.csv')
                gt = pd.read_csv(datadir + '/train.csv')
                valid = pd.read_csv(datadir + '/valid.csv')
                test = pd.read_csv(datadir + '/test.csv')

                test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])[:samples]
                train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'],
                                         robust=robust)

                save_path = 'models/' + model_name + '/' + dir
                if robust:
                    save_path = save_path + '_robust'

                os.makedirs(save_path, exist_ok=True)
                save_path = save_path

                try:
                    logging.info('loading model from {}', save_path)
                    model.load(save_path)
                except:
                    logging.info('training model')

                    valid_df = merge_sources(valid, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
                    model.train(train_df, valid_df, dir)

                    precision, recall, fmeasure = model.evaluation(test_df)

                    text_file = open(save_path + '_report.txt', "w")
                    text_file.write('p:' + str(precision) + ', r:' + str(recall) + ', f1:' + str(fmeasure))
                    text_file.close()
                    model.save(save_path)

                def predict_fn(x, **kwargs):
                    return model.predict(x, **kwargs)

                def predict_fn_mojito(x):
                    return model.predict(x, mojito=True)

                examples = pd.DataFrame()
                certas = pd.DataFrame()
                for i in range(len(test_df)):
                    rand_row = test_df.iloc[i]
                    l_id = int(rand_row['ltable_id'])
                    l_tuple = lsource.iloc[l_id]
                    r_id = int(rand_row['rtable_id'])
                    r_tuple = rsource.iloc[r_id]

                    prediction = get_original_prediction(l_tuple, r_tuple, predict_fn)
                    class_to_explain = np.argmax(prediction)

                    label = rand_row["label"]
                    row_id = str(l_id) + '-' + str(r_id)
                    item = get_row(l_tuple, r_tuple)

                    try:
                        # CERTA
                        print('certa')
                        num_triangles = 100

                        t0 = time.perf_counter()

                        saliency_df, cf_summary, counterfactual_examples, triangles = explain(l_tuple, r_tuple, lsource,
                                                                                              rsource, predict_fn,
                                                                                              datadir,
                                                                                              num_triangles=num_triangles,
                                                                                              fast=fast,
                                                                                              max_predict=max_predict)

                        latency_c = time.perf_counter() - t0

                        certa_row = {'summary': cf_summary, 'type': 'certa', 'latency': latency_c,
                                     'match': class_to_explain,
                                     'label': label, 'row': row_id, 'prediction': prediction}

                        certas = certas.append(certa_row, ignore_index=True)
                        cf_dir = exp_dir + dir + '/' + model_name + '/' + str(i)
                        os.makedirs(cf_dir, exist_ok=True)
                        counterfactual_examples.to_csv(cf_dir + '/certa.csv')

                        item['match'] = prediction[1]
                        item['label'] = label
                        examples = examples.append(item, ignore_index=True)
                        print(item)
                        print(i)
                    except:
                        print(traceback.format_exc())
                        print(f'skipped item {str(i)}')
                        item.head()

                certas.to_csv(exp_dir + dir + '/' + model_name + '/certa.csv')
                examples.to_csv(exp_dir + dir + '/' + model_name + '/examples.csv')


import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    samples = 50
    type = 'deeper'
    filtered_datasets = ['dirty_dblp_scholar', 'dirty_amazon_itunes', 'dirty_walmart_amazon', 'dirty_dblp_acm',
                         'abt_buy', 'fodo_zaga', 'beers',
                         'amazon_google', 'itunes_amazon', 'walmart_amazon',
                         'dblp_scholar', 'dblp_acm']
    model = from_type(type)
    evaluate(model, samples=samples, filtered_datasets=filtered_datasets, max_predict=-1, fast=True)
