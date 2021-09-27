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
experiments_dir = 'quantitative/'


def evaluate(model: ERModel, samples: int = 50, filtered_datasets: list = [], exp_dir: str = experiments_dir,
             fast: bool = False, max_predict: int = -1):
    if not exp_dir.endswith('/'):
        exp_dir = exp_dir + '/'

    for subdir, dirs, files in os.walk(root_datadir):
        for dir in dirs:
            if dir not in filtered_datasets:
                continue
            for robust in [False, True]:
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

                mojito = Mojito(test_df.columns,
                                attr_to_copy='left',
                                split_expression=" ",
                                class_names=['no_match', 'match'],
                                lprefix='', rprefix='',
                                feature_selection='lasso_path')

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

                landmark_explainer = Landmark(lambda x: predict_fn(x)['match_score'].values, test_df, lprefix='',
                                              exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], rprefix='',
                                              split_expression=r' ')

                shap_explainer = shap.KernelExplainer(lambda x: predict_fn(x)['match_score'].values,
                                                      train_df.drop(['label'], axis=1).astype(str)[:50], link='identity')

                examples = pd.DataFrame()
                certas = pd.DataFrame()
                landmarks = pd.DataFrame()
                shaps = pd.DataFrame()
                mojitos_c = pd.DataFrame()
                mojitos_d = pd.DataFrame()
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

                        saliency_df, cf_summary, counterfactual_examples, triangles = explain(l_tuple, r_tuple, lsource[:2*max_predict],
                                                                                              rsource[:2*max_predict], predict_fn, datadir,
                                                                                              num_triangles=num_triangles,
                                                                                              fast=fast, max_predict=max_predict)

                        latency_c = time.perf_counter() - t0

                        certa_saliency = saliency_df.transpose().to_dict()[0]
                        certa_row = {'explanation': certa_saliency, 'type': 'certa', 'latency': latency_c,
                                     'match': class_to_explain,
                                     'label': label, 'row': row_id, 'prediction': prediction}

                        certas = certas.append(certa_row, ignore_index=True)


                        # Mojito
                        print('mojito')
                        t0 = time.perf_counter()
                        mojito_exp_copy = mojito.copy(predict_fn_mojito, item,
                                                      num_features=15,
                                                      num_perturbation=100)

                        latency_m = time.perf_counter() - t0

                        mojito_exp = mojito_exp_copy.groupby('attribute')['weight'].mean().to_dict()

                        if 'id' in mojito_exp:
                            mojito_exp.pop('id', None)


                        mojito_row = {'explanation': mojito_exp, 'type': 'mojito-c', 'latency': latency_m,
                                   'match': class_to_explain,
                                   'label': label, 'row': row_id, 'prediction': prediction}
                        mojitos_c = mojitos_c.append(mojito_row, ignore_index=True)

                        t0 = time.perf_counter()
                        mojito_exp_drop = mojito.drop(predict_fn_mojito, item,
                                                      num_features=15,
                                                      num_perturbation=100)

                        latency_m = time.perf_counter() - t0

                        mojito_exp = mojito_exp_drop.groupby('attribute')['weight'].mean().to_dict()

                        if 'id' in mojito_exp:
                            mojito_exp.pop('id', None)

                        mojito_row = {'explanation': mojito_exp, 'type': 'mojito-d', 'latency': latency_m,
                                      'match': class_to_explain,
                                      'label': label, 'row': row_id, 'prediction': prediction}
                        mojitos_d = mojitos_d.append(mojito_row, ignore_index=True)

                        # landmark
                        print('landmark')
                        labelled_item = item.copy()
                        labelled_item['label'] = int(label)
                        labelled_item['id'] = i

                        t0 = time.perf_counter()
                        land_explanation = landmark_explainer.explain(labelled_item)
                        latency_l = time.perf_counter() - t0

                        land_exp = land_explanation.groupby('column')['impact'].sum().to_dict()

                        land_row = {'explanation': str(land_exp), 'type': 'landmark', 'latency': latency_l,
                                   'match': class_to_explain,
                                   'label': label, 'row': row_id, 'prediction': prediction}
                        landmarks = landmarks.append(land_row, ignore_index=True)

                        # SHAP
                        print('shap')
                        shap_instance = test_df.iloc[i, 1:].drop(['ltable_id', 'rtable_id']).astype(str)

                        t0 = time.perf_counter()
                        shap_values = shap_explainer.shap_values(shap_instance, nsamples=100)

                        latency_s = time.perf_counter() - t0

                        match_shap_values = shap_values

                        shap_saliency = dict()
                        for sv in range(len(match_shap_values)):
                            shap_saliency[train_df.columns[1 + sv]] = match_shap_values[sv]

                        shap_row = {'explanation': str(shap_saliency), 'type': 'shap', 'latency': latency_s,
                                    'match': class_to_explain,
                                    'label': label, 'row': row_id, 'prediction': prediction}
                        shaps = shaps.append(shap_row, ignore_index=True)

                        item['match'] = prediction[1]
                        examples = examples.append(item, ignore_index=True)
                        print(item)
                        print(i)
                    except:
                        print(traceback.format_exc())
                        print(f'skipped item {str(i)}')
                        item.head()


                mojitos_d.to_csv(exp_dir + dir + '/' + model_name + '/mojito_d.csv')
                mojitos_c.to_csv(exp_dir + dir + '/' + model_name + '/mojito_c.csv')
                landmarks.to_csv(exp_dir + dir + '/' + model_name + '/landmark.csv')
                certas.to_csv(exp_dir + dir + '/' + model_name + '/certa.csv')
                shaps.to_csv(exp_dir + dir + '/' + model_name + '/shap.csv')
                examples.to_csv(exp_dir + dir + '/' + model_name + '/examples.csv')



import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    samples = 50
    type = 'deeper'
    filtered_datasets = []
    model = from_type(type)
    evaluate(model, samples=samples, filtered_datasets=filtered_datasets, max_predict=3000, fast=True)
