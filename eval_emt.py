import logging

import pandas as pd
import numpy as np
import os
from certa.local_explain import dataset_local, get_original_prediction
from certa.triangles_method import explainSamples
from certa.eval import expl_eval
from certa.utils import merge_sources
from models.bert import EMTERModel


def predict_fn(x, model, ignore_columns=['ltable_id', 'rtable_id', 'label']):
    return model.predict(x)


root_datadir = 'datasets/'
experiments_dir = 'experiments_sep/'
generate_cf = False

def eval_emt(samples=50, max_predict = 500, discard_bad = False, filtered_datasets: list = [],
             exp_dir=experiments_dir):
    if not exp_dir.endswith('/'):
        exp_dir = exp_dir + '/'
    evals_list = []
    for subdir, dirs, files in os.walk(root_datadir):
        for dir in dirs:
            if dir in filtered_datasets:
                continue
            for robust in [False, True]:
                os.makedirs(exp_dir + dir, exist_ok=True)
                model_name = 'emt'
                if robust:
                    model_name = model_name + '_robust'
                os.makedirs(exp_dir + dir + '/' + model_name, exist_ok=True)
                if dir == 'temporary':
                    continue
                logging.info(f'working on {dir}')
                datadir = os.path.join(root_datadir, dir)
                logging.info(f'reading data from {datadir}')

                lsource = pd.read_csv(datadir + '/tableA.csv')
                rsource = pd.read_csv(datadir + '/tableB.csv')
                gt = pd.read_csv(datadir + '/train.csv')
                valid = pd.read_csv(datadir + '/valid.csv')
                test = pd.read_csv(datadir + '/test.csv')

                test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], []).dropna()[:samples]

                os.makedirs('models/' + model_name + '/' + dir, exist_ok=True)
                save_path = 'models/' + model_name + '/' + dir
                if robust:
                    save_path = save_path + '_robust'
                model = EMTERModel()
                try:
                    logging.info(f'loading model from {save_path}')
                    model.load(save_path)
                except:
                    logging.info('training model')
                    train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'], robust=robust).dropna()
                    valid_df = merge_sources(valid, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id']).dropna()
                    report = model.classic_training(train_df, valid_df, dir)
                    text_file = open(save_path + '_report.txt', "w")
                    text_file.write(str(report))
                    text_file.close()
                    model.save(save_path)

                tmin = 0.5
                tmax = 0.5

                evals = pd.DataFrame()
                cf_evals = pd.DataFrame()
                for i in range(len(test_df)):
                    rand_row = test_df.iloc[i]
                    l_id = int(rand_row['ltable_id'])
                    l_tuple = lsource.iloc[l_id]
                    r_id = int(rand_row['rtable_id'])
                    r_tuple = rsource.iloc[r_id]

                    prediction = get_original_prediction(l_tuple, r_tuple, model, predict_fn)
                    class_to_explain = np.argmax(prediction)

                    label = rand_row["label"]

                    # get triangle 'cuts' depending on the length of the sources
                    up_bound = min(len(lsource), len(rsource))
                    cuts = [100]

                    for nt in cuts:
                        local_samples, gleft_df, gright_df = dataset_local(l_tuple, r_tuple, model, lsource, rsource, datadir,
                                                                            tmin, tmax, predict_fn, num_triangles=nt,
                                                                            class_to_explain=class_to_explain, use_predict=True,
                                                                            max_predict=max_predict)
                        if len(local_samples) > 2:
                            maxLenAttributeSet = len(l_tuple) - 1
                            explanation, flipped_pred, triangles = explainSamples(local_samples,
                                                                                  [pd.concat([lsource, gright_df]),
                                                                                   pd.concat([rsource, gleft_df])],
                                                                                  model, predict_fn, class_to_explain,
                                                                                  maxLenAttributeSet=maxLenAttributeSet,
                                                                                  check=True,
                                                                                  discard_bad=discard_bad)
                            triangles_df = pd.DataFrame()
                            if len(triangles) > 0:
                                triangles_df = pd.DataFrame(triangles)
                                triangles_df.to_csv(
                                    exp_dir + dir + '/' + model_name + '/tri_' + str(l_id) + '-' + str(r_id) + '_' + str(
                                        nt) + '_' + str(tmin) + '-' + str(tmax) + '.csv')
                            for exp in explanation:
                                e_attrs = exp.split('/')
                                e_score = explanation[exp]
                                expl_evaluation = expl_eval(class_to_explain, e_attrs, e_score, lsource, l_tuple, model,
                                                            prediction, rsource,
                                                            r_tuple, predict_fn)
                                expl_evaluation['t_requested'] = nt
                                expl_evaluation['t_obtained'] = len(triangles)
                                expl_evaluation['label'] = label
                                identity = triangles_df[3].apply(lambda x: int(x)).sum()
                                expl_evaluation['identity'] = identity
                                symmetry = triangles_df[4].apply(lambda x: int(x)).sum()
                                expl_evaluation['symmetry'] = symmetry
                                n_good = triangles_df[5].apply(lambda x: int(x)).sum()
                                expl_evaluation['t_good'] = n_good
                                expl_evaluation['t_bad'] = len(triangles_df) - n_good

                                evals = evals.append(expl_evaluation, ignore_index=True)
                                evals.to_csv(exp_dir + dir + '/' + model_name + '/eval.csv')

                            if generate_cf:
                                try:
                                    cf_class = abs(1 - int(class_to_explain))

                                    local_samples_cf = dataset_local(l_tuple, r_tuple, model, lsource, rsource, datadir, tmin,
                                                                     tmax, predict_fn,
                                                                     num_triangles=nt, class_to_explain=cf_class)

                                    if len(local_samples_cf) > 2:
                                        explanation_cf, flipped_pred_cf, triangles_cf = explainSamples(local_samples,
                                                                                                       [lsource, rsource],
                                                                                                       model, predict_fn,
                                                                                                       cf_class,
                                                                                                       maxLenAttributeSet, True)
                                        for exp_cf in explanation_cf:
                                            e_attrs = exp_cf.split('/')
                                            e_score = explanation_cf[exp_cf]
                                            cf_expl_evaluation = expl_eval(class_to_explain, e_attrs, e_score, lsource, l_tuple,
                                                                           model, prediction, rsource,
                                                                           r_tuple, predict_fn)
                                            cf_expl_evaluation['t_requested'] = nt
                                            cf_expl_evaluation['t_obtained'] = len(triangles_cf)
                                            cf_expl_evaluation['label'] = label
                                            cf_evals = cf_evals.append(cf_expl_evaluation, ignore_index=True)
                                            cf_evals.to_csv(exp_dir + dir + '/' + model_name + '/eval-cf.csv')
                                        if len(triangles_cf) > 0:
                                            pd.DataFrame(triangles_cf).to_csv(
                                                exp_dir + dir + '/' + model_name + '/tri_cf_' + str(l_id) + '-' + str(r_id) + '_' + str(
                                                    nt) + '_' + str(
                                                    tmin) + '-' + str(tmax) + '.csv')
                                except:
                                    pass
                evals.to_csv(exp_dir + dir + "/"+ model_name +"/eval_" + str(tmin) + '-' + str(tmax) + '.csv')
                evals_list.append(evals)
                if generate_cf:
                    cf_evals.to_csv(exp_dir + dir + "/"+ model_name +"/eval_cf_" + str(tmin) + '-' + str(tmax) + '.csv')
    return evals_list
