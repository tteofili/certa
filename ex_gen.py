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

from models.utils import from_type

root_datadir = 'datasets/'
experiments_dir = 'examples/'


def evaluate(model: str, samples: int = -1, filtered_datasets: list = [], exp_dir: str = experiments_dir,
             fast: bool = False, max_predict: int = -1):
    if not exp_dir.endswith('/'):
        exp_dir = exp_dir + '/'

    for subdir, dirs, files in os.walk(root_datadir):
        for dir in dirs:
            if dir not in filtered_datasets:
                continue
            model = from_type(mtype)

            os.makedirs(exp_dir + dir, exist_ok=True)
            model_name = mtype
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
            train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])

            k = 3

            mojito = Mojito(test_df.columns,
                            attr_to_copy='left',
                            split_expression=" ",
                            class_names=['no_match', 'match'],
                            lprefix='', rprefix='',
                            feature_selection='lasso_path')

            save_path = 'models/' + model_name + '/' + dir

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
            mojitos = pd.DataFrame()
            cf = pd.DataFrame()

            for i in range(len(test_df)):
                rand_row = test_df.iloc[i]
                l_id = int(rand_row['ltable_id'])
                l_tuple = lsource.iloc[l_id]
                r_id = int(rand_row['rtable_id'])
                r_tuple = rsource.iloc[r_id]

                prediction = get_original_prediction(l_tuple, r_tuple, predict_fn)
                class_to_explain = np.argmax(prediction)

                label = rand_row["label"]
                if class_to_explain != label:
                    row_id = str(l_id) + '-' + str(r_id)
                    item = get_row(l_tuple, r_tuple)

                    try:
                        # Mojito
                        print('mojito')

                        if class_to_explain == 0:

                            t0 = time.perf_counter()
                            mojito_exp_copy = mojito.copy(predict_fn_mojito, item,
                                                          num_features=15,
                                                          num_perturbation=100)

                            latency_m = time.perf_counter() - t0

                            mojito_exp = mojito_exp_copy.groupby('attribute')['weight'].mean().to_dict()

                            if 'id' in mojito_exp:
                                mojito_exp.pop('id', None)


                        else:
                            t0 = time.perf_counter()
                            mojito_exp_drop = mojito.drop(predict_fn_mojito, item,
                                                          num_features=15,
                                                          num_perturbation=100)

                            latency_m = time.perf_counter() - t0

                            mojito_exp = mojito_exp_drop.groupby('attribute')['weight'].mean().to_dict()

                            if 'id' in mojito_exp:
                                mojito_exp.pop('id', None)

                        mojito_exp['type'] = 'mojito'
                        check, effect_eval = check_saliency(model, l_tuple, r_tuple, predict_fn, mojito_exp, k, prediction[1])
                        if check:
                            continue
                        mojito_row = {'explanation': mojito_exp, 'type': 'mojito-d', 'latency': latency_m,
                                      'match': class_to_explain,
                                      'label': label, 'row': row_id, 'prediction': prediction,
                                      'score_drop': effect_eval['score_drop'], 'score_copy': effect_eval['score_copy']}

                        # landmark
                        print('landmark')
                        labelled_item = item.copy()
                        labelled_item['label'] = int(label)
                        labelled_item['id'] = i

                        t0 = time.perf_counter()
                        land_explanation = landmark_explainer.explain(labelled_item)
                        latency_l = time.perf_counter() - t0

                        land_exp = land_explanation.groupby('column')['impact'].sum().to_dict()


                        land_exp['type'] = 'landmark'
                        check, effect_eval = check_saliency(model, l_tuple, r_tuple, predict_fn, land_exp, k, prediction[1])
                        if check:
                            continue
                        land_row = {'explanation': str(land_exp), 'type': 'landmark', 'latency': latency_l,
                                    'match': class_to_explain,
                                    'label': label, 'row': row_id, 'prediction': prediction,
                                    'score_drop': effect_eval['score_drop'], 'score_copy': effect_eval['score_copy']}

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


                        shap_saliency['type'] = 'shap'
                        check, effect_eval = check_saliency(model, l_tuple, r_tuple, predict_fn, shap_saliency, k, prediction[1])
                        if check:
                            continue
                        shap_row = {'explanation': str(shap_saliency), 'type': 'shap', 'latency': latency_s,
                                    'match': class_to_explain,
                                    'label': label, 'row': row_id, 'prediction': prediction,
                                    'score_drop': effect_eval['score_drop'], 'score_copy': effect_eval['score_copy']}


                        # CERTA
                        print('certa')
                        num_triangles = 100

                        t0 = time.perf_counter()

                        saliency_df, cf_summary, counterfactual_examples, triangles = explain(l_tuple, r_tuple, lsource,
                                                                                              rsource, predict_fn, datadir,
                                                                                              num_triangles=num_triangles,
                                                                                              fast=fast, max_predict=max_predict,
                                                                                              token_parts=True)

                        latency_c = time.perf_counter() - t0

                        if len(saliency_df) == 0:
                            continue

                        certa_saliency = saliency_df.transpose().to_dict()[0]

                        certa_saliency['type'] = 'certa'
                        check, effect_eval = check_saliency(model, l_tuple, r_tuple, predict_fn, certa_saliency, k,
                                                            prediction[1])
                        
                        certa_row = {'explanation': certa_saliency, 'type': 'certa', 'latency': latency_c,
                                     'match': class_to_explain,
                                     'label': label, 'row': row_id, 'prediction': prediction,
                                     'score_drop': effect_eval['score_drop'], 'score_copy': effect_eval['score_copy']}

                        item['match'] = prediction[1]
                        item['label'] = label
                        item['cf_summary'] = str(cf_summary.to_dict())

                        mojitos = mojitos.append(mojito_row, ignore_index=True)
                        landmarks = landmarks.append(land_row, ignore_index=True)
                        shaps = shaps.append(shap_row, ignore_index=True)
                        certas = certas.append(certa_row, ignore_index=True)
                        examples = examples.append(item, ignore_index=True)
                        cf = cf.append(counterfactual_examples[:1], ignore_index=True)

                        print(item)
                        print(i)
                    except:
                        print(traceback.format_exc())
                        print(f'skipped item {str(i)}')
                        item.head()


            mojitos.to_csv(exp_dir + dir + '/' + model_name + '/mojito.csv')
            landmarks.to_csv(exp_dir + dir + '/' + model_name + '/landmark.csv')
            shaps.to_csv(exp_dir + dir + '/' + model_name + '/shap.csv')
            examples.to_csv(exp_dir + dir + '/' + model_name + '/examples.csv')
            certas.to_csv(exp_dir + dir + '/' + model_name + '/certa.csv')
            cf.to_csv(exp_dir + dir + '/' + model_name + '/cf.csv')


def check_saliency(model, l_tuple, r_tuple, predict_fn, saliency, top_k, score):
    check = False
    lprefix = 'ltable_'
    rprefix = 'rtable_'
    if score > 0.5:
        orig_class = 1
    else:
        orig_class = 0
    eval_series = pd.Series()
    # eval saliencies
    print(saliency)
    # get top k important attributes
    score_drops = []
    class_drops = []
    score_copies = []
    class_copies = []
    for k in range(1, top_k):
        saliency_c = saliency.copy()
        exp_type = saliency_c['type']
        print(saliency_c.pop('type'))
        if exp_type == 'certa':
            explanation_attributes = sorted(saliency_c, key=saliency_c.get, reverse=True)[:k]
        elif score < 0.5:
            saliency_c = {k: v for k, v in saliency_c.items() if v < 0}
            explanation_attributes = sorted(saliency_c, key=saliency_c.get)[:k]
        else:
            saliency_c = {k: v for k, v in saliency_c.items() if v > 0}
            explanation_attributes = sorted(saliency_c, key=saliency_c.get, reverse=True)[:k]

        # change those attributes
        try:
            lt = l_tuple.copy()
            rt = r_tuple.copy()
            modified_row = get_row(lt, rt)
            for e in explanation_attributes:
                modified_row[e] = ''
            modified_tuple_prediction = predict_fn(modified_row)[['nomatch_score', 'match_score']].values[0]
            score_drop = modified_tuple_prediction[1]
            class_drop = np.argmax(modified_tuple_prediction)
            eval_series['top_' + exp_type + '_' + model.name] = explanation_attributes
            score_drops.append(score_drop)
            class_drops.append(class_drop)
            if class_drop != orig_class:
                check = True
        except Exception as e:
            print(traceback.format_exc())
        try:
            lt = l_tuple.copy()
            rt = r_tuple.copy()
            modified_row = get_row(lt, rt)
            for e in explanation_attributes:
                if e.startswith(lprefix):
                    new_e = e.replace(lprefix, rprefix)
                else:
                    new_e = e.replace(rprefix, lprefix)
                modified_row[e] = modified_row[new_e]
            modified_tuple_prediction = predict_fn(modified_row)[['nomatch_score', 'match_score']].values[0]
            score_copy = modified_tuple_prediction[1]
            class_copy = np.argmax(modified_tuple_prediction)
            score_copies.append(score_copy)
            class_copies.append(class_copy)
            if not check and class_copy != orig_class:
                check = True
        except Exception as e:
            print(traceback.format_exc())
    eval_series['score_drop'] = score_drops
    eval_series['class_drop'] = class_drops
    eval_series['score_copy'] = score_copies
    eval_series['class_copy'] = class_copies
    return check, eval_series


import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    samples = 500
    mtype = 'emt'
    filtered_datasets = ['abt_buy']
    evaluate(mtype, samples=samples, filtered_datasets=filtered_datasets, max_predict=300, fast=True)
