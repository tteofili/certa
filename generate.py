import argparse
import os
import time
import traceback

import dice_ml
import numpy as np
import pandas as pd
import shap

from baselines.landmark import Landmark
from baselines.lime_c import LimeCounterfactual
from baselines.mojito import Mojito
from baselines.shap_c import ShapCounterfactual
from certa.explain import CertaExplainer
from certa.local_explain import get_original_prediction, get_row
from certa.utils import merge_sources
from certa.models.utils import get_model

experiments_dir = 'experiments/'
base_datadir = 'datasets/'


def generate(mtype: str, samples: int = -1, filtered_datasets: list = [], exp_dir: str = experiments_dir,
             compare=False, da=None):
    if not exp_dir.endswith('/'):
        exp_dir = exp_dir + '/'
    exp_dir = exp_dir + 'generated/'
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

        generate_all(compare, dataset, exp_dir, lsource, model, model_name, predict_fn, predict_fn_mojito,
                          rsource, test_df, train_df, da)


def generate_all(compare, dataset, exp_dir, lsource, model, model_name, predict_fn, predict_fn_mojito, rsource,
                  test_df, train_df, da):
    train_noids = train_df.copy().astype(str)
    if 'ltable_id' in train_noids.columns and 'rtable_id' in train_noids.columns:
        train_noids = train_df.drop(['ltable_id', 'rtable_id'], axis=1)
    certa_explainer = CertaExplainer(lsource, rsource, data_augmentation=da)
    if compare:
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
        landmarks = pd.DataFrame()
        shaps = pd.DataFrame()
        mojitos = pd.DataFrame()

    examples = pd.DataFrame()
    certas = pd.DataFrame()
    for idx in range(len(test_df)):
        rand_row = test_df.iloc[idx]
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
            cf_dir = exp_dir + dataset + '/' + model_name + '/' + str(idx)
            os.makedirs(cf_dir, exist_ok=True)
            dest_file = cf_dir + '/certa.csv'
            if not os.path.exists(dest_file):

                # CERTA
                print('certa')
                t0 = time.perf_counter()
                certa_saliency = None
                num_triangles = 10
                while certa_saliency is None:
                    try:
                        saliency_df, cf_summary, cf_ex, triangles, lattices = certa_explainer.explain(l_tuple, r_tuple, predict_fn,
                                                                                              debug=True, num_triangles=num_triangles)


                        latency_c = time.perf_counter() - t0

                        certa_saliency = saliency_df.transpose().to_dict()[0]
                    except:
                        pass
                    num_triangles += 50
                    if num_triangles > 200:
                        break
                certa_row = {'explanation': certa_saliency, 'summary' : cf_summary.to_dict(), 'type': 'certa', 'latency': latency_c,
                             'match': class_to_explain,
                             'label': label, 'row': row_id, 'prediction': prediction}

                cf_ex.to_csv(dest_file)
                lidx = 0
                for lattice in lattices:
                    lattice.triangle.to_csv(cf_dir + '/triangle_' + str(lidx) + '.csv')
                    dot_lattice = lattice.hasse()
                    with open(cf_dir + '/lattice_' + str(lidx) + '.dot', 'w') as f:
                        f.write(dot_lattice)
                    lidx += 1

                certas = certas.append(certa_row, ignore_index=True)

                if compare:
                    instance = pd.DataFrame(rand_row).transpose().astype(str)
                    for c in ['label', 'ltable_id', 'rtable_id']:
                        if c in instance.columns:
                            instance = instance.drop([c], axis=1)

                    if not os.path.exists(cf_dir + '/limec.csv'):
                        print('lime-c')
                        try:
                            limec_explainer = LimeCounterfactual(model, predict_fn_mojito, None, 0.5,
                                                                 train_noids.drop(['label'], axis=1).columns, max_features=6,
                                                                 time_maximum=300)
                            limec_exp = limec_explainer.explanation(instance)
                            print(limec_exp)
                            if limec_exp is not None:
                                limec_exp['cf_example'].to_csv(cf_dir + '/limec.csv')
                        except:
                            print(traceback.format_exc())
                            print(f'skipped item {str(idx)}')
                            pass

                    if not os.path.exists(cf_dir + '/shapc.csv'):
                        print('shap-c')
                        try:
                            shapc_explainer = ShapCounterfactual(lambda x: predict_fn(x)[['nomatch_score','match_score']].values, 0.5,
                                                                 train_noids.drop(['label'], axis=1).columns, time_maximum=300, max_features=6)

                            sc_exp = shapc_explainer.explanation(instance, train_noids.drop(['label'], axis=1)[:50])
                            print(f'{idx}- shap-c:{sc_exp}')
                            if sc_exp is not None:
                                sc_exp['cf_example'].to_csv(cf_dir + '/shapc.csv')
                        except:
                            print(traceback.format_exc())
                            print(f'skipped item {str(idx)}')
                            pass

                    if not os.path.exists(cf_dir + '/dice_random.csv'):
                        print('dice_r')
                        try:

                            d = dice_ml.Data(dataframe=test_df.drop(['ltable_id', 'rtable_id'], axis=1),
                                             continuous_features=[],
                                             outcome_name='label')
                            # random
                            m = dice_ml.Model(model=model, backend='sklearn')
                            exp = dice_ml.Dice(d, m, method='random')
                            dice_exp = exp.generate_counterfactuals(instance,
                                                                    total_CFs=10, desired_class="opposite")
                            dice_exp_df = dice_exp.cf_examples_list[0].final_cfs_df
                            print(f'random:{idx}:{dice_exp_df}')
                            if dice_exp_df is not None:
                                dice_exp_df.to_csv(cf_dir + '/dice_random.csv')
                        except:
                            print(traceback.format_exc())
                            print(f'skipped item {str(idx)}')
                            pass

                    # Mojito
                    print('mojito')
                    if class_to_explain == 1:
                        t0 = time.perf_counter()
                        mojito_exp_drop = mojito.drop(predict_fn_mojito, item,
                                                      num_features=15,
                                                      num_perturbation=100)

                        latency_m = time.perf_counter() - t0

                        mojito_exp = mojito_exp_drop.groupby('attribute')['weight'].mean().to_dict()
                    else:
                        t0 = time.perf_counter()
                        mojito_exp_copy = mojito.copy(predict_fn_mojito, item,
                                                      num_features=15,
                                                      num_perturbation=100)

                        latency_m = time.perf_counter() - t0

                        mojito_exp = mojito_exp_copy.groupby('attribute')['weight'].mean().to_dict()

                    if 'id' in mojito_exp:
                        mojito_exp.pop('id', None)

                    mojito_row = {'explanation': mojito_exp, 'type': 'mojito', 'latency': latency_m,
                                  'match': class_to_explain,
                                  'label': label, 'row': row_id, 'prediction': prediction}
                    mojitos = mojitos.append(mojito_row, ignore_index=True)

                    # landmark
                    print('landmark')
                    labelled_item = item.copy()
                    labelled_item['label'] = int(label)
                    labelled_item['id'] = idx

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
                    shap_instance = test_df.iloc[idx, 1:].drop(['ltable_id', 'rtable_id']).astype(str)

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
            print(idx)
        except:
            print(traceback.format_exc())
            print(f'skipped item {str(idx)}')
            item.head()
    os.makedirs(exp_dir + dataset + '/' + model_name, exist_ok=True)
    if compare:
        mojitos.to_csv(exp_dir + dataset + '/' + model_name + '/mojito.csv')
        landmarks.to_csv(exp_dir + dataset + '/' + model_name + '/landmark.csv')
        shaps.to_csv(exp_dir + dataset + '/' + model_name + '/shap.csv')
    examples.to_csv(exp_dir + dataset + '/' + model_name + '/examples.csv')
    certas.to_csv(exp_dir + dataset + '/' + model_name + '/certa.csv')


import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run saliency experiments.')
    parser.add_argument('--base_dir', metavar='b', type=str, help='the datasets base directory',
                        required=True)
    parser.add_argument('--model_type', metavar='m', type=str, help='the ER model type to evaluate',
                        choices=['dm', 'deeper', 'ditto'], required=True)
    parser.add_argument('--datasets', metavar='d', type=str, nargs='+', required=True,
                        help='the datasets to be used for the evaluation')
    parser.add_argument('--samples', metavar='s', type=int, default=-1,
                        help='no. of samples from the test set used for the evaluation')
    parser.add_argument('--compare', metavar='c', type=bool, default=False,
                        help='whether comparing CERTA with baselines')
    parser.add_argument('--da', metavar='da', type=str, default='on_demand',
                        help='whether enabling CERTA data-augmentation feature')

    args = parser.parse_args()
    base_datadir = args.base_dir
    if not base_datadir.endswith('/'):
        base_datadir = base_datadir + '/'
    filtered_datasets = args.datasets
    mtype = args.model_type
    samples = args.samples
    compare = args.compare
    da = args.da

    generate(mtype, filtered_datasets=filtered_datasets, samples=samples, compare=compare, da=da)