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

from metrics.saliency import get_faithfullness
from models.utils import get_model

experiments_dir = 'experiments/'
base_datadir = 'datasets/'

def evaluate(mtype: str, samples: int = 5, filtered_datasets: list = [], exp_dir: str = experiments_dir,
             compare=True):
    if not exp_dir.endswith('/'):
        exp_dir = exp_dir + '/'
    os.makedirs(exp_dir, exist_ok=True)
    for dataset in filtered_datasets:
        os.makedirs(exp_dir + dataset, exist_ok=True)
        modeldir = 'models/saved/' + mtype + '/' + dataset
        model_name = mtype
        datadir = base_datadir + dataset
        model = get_model(mtype, modeldir, datadir, dataset)
        def predict_fn(x, **kwargs):
            return model.predict(x, **kwargs)

        def predict_fn_mojito(x):
            return model.predict(x, mojito=True)

        test = pd.read_csv(datadir + '/test.csv')
        lsource = pd.read_csv(datadir + '/tableA.csv')
        rsource = pd.read_csv(datadir + '/tableB.csv')
        gt = pd.read_csv(datadir + '/train.csv')
        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])[:samples]
        train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])
        mojito = Mojito(test_df.columns,
                        attr_to_copy='left',
                        split_expression=" ",
                        class_names=['no_match', 'match'],
                        lprefix='', rprefix='',
                        feature_selection='lasso_path')
        landmark_explainer = Landmark(lambda x: predict_fn(x)['match_score'].values, test_df, lprefix='',
                                      exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], rprefix='',
                                      split_expression=r' ')

        shap_explainer = shap.KernelExplainer(lambda x: predict_fn(x)['match_score'].values,
                                              train_df.drop(['label'], axis=1).astype(str)[:100], link='identity')

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

                saliency_df, cf_summary, counterfactual_examples, triangles = explain(l_tuple, r_tuple, lsource,
                                                                                      rsource, predict_fn, datadir,
                                                                                      num_triangles=num_triangles)

                latency_c = time.perf_counter() - t0

                certa_saliency = saliency_df.transpose().to_dict()[0]
                certa_row = {'explanation': certa_saliency, 'type': 'certa', 'latency': latency_c,
                             'match': class_to_explain,
                             'label': label, 'row': row_id, 'prediction': prediction}

                certas = certas.append(certa_row, ignore_index=True)

                if compare:
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
                    shap_values = shap_explainer.shap_values(shap_instance, nsamples=10)

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
                item['label'] = label
                examples = examples.append(item, ignore_index=True)
                print(item)
                print(i)
            except:
                print(traceback.format_exc())
                print(f'skipped item {str(i)}')
                item.head()

        if compare:
            mojitos_d.to_csv(exp_dir + dataset + '/' + model_name + '/mojito_d.csv')
            mojitos_c.to_csv(exp_dir + dataset + '/' + model_name + '/mojito_c.csv')
            landmarks.to_csv(exp_dir + dataset + '/' + model_name + '/landmark.csv')
            shaps.to_csv(exp_dir + dataset + '/' + model_name + '/shap.csv')
            examples.to_csv(exp_dir + dataset + '/' + model_name + '/examples.csv')
        certas.to_csv(exp_dir + dataset + '/' + model_name + '/certa.csv')
        faithfulness = get_faithfullness(model, '%s%s%s/%s' % ('', experiments_dir, dataset, mtype),
                                         test_df)
        print(f'{mtype}: faithfulness for {dataset}: {faithfulness}')


import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    mtype = 'dm'
    filtered_datasets = ['fodo_zaga']
    evaluate(mtype, filtered_datasets=filtered_datasets)
