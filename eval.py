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
from certa.utils import merge_sources, to_token_df
from certa.metrics.counterfactual import get_validity, get_proximity, get_sparsity, get_diversity
from certa.metrics.saliency import get_faithfulness, get_confidence
from certa.models.utils import get_model

experiments_dir = 'experiments/'
base_datadir = 'datasets/'


def eval_all(compare, dataset, exp_dir, lsource, model, model_name, mtype, predict_fn, predict_fn_mojito, rsource,
             test_df, train_df, da, num_triangles, token, eval_only, predict_fn_c, predict_fn_t):
    certa_explainer = CertaExplainer(lsource, rsource, data_augmentation=da)
    if compare:
        train_noids = train_df.copy().astype(str)
        if 'ltable_id' in train_noids.columns and 'rtable_id' in train_noids.columns:
            train_noids = train_df.drop(['ltable_id', 'rtable_id'], axis=1)

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

        limec_explainer = LimeCounterfactual(model, predict_fn_c, None, 0.5, train_noids.columns, time_maximum=300,
                                             class_names=['nomatch_score', 'match_score'])

        shapc_explainer = ShapCounterfactual(predict_fn_c, 0.5, train_noids.columns, time_maximum=300)

        d = dice_ml.Data(dataframe=test_df.drop(['ltable_id', 'rtable_id'], axis=1),
                         continuous_features=[],
                         outcome_name='label')
        # random
        m = dice_ml.Model(model=model, backend='sklearn')
        exp = dice_ml.Dice(d, m, method='random')

        landmarks = pd.DataFrame()
        shaps = pd.DataFrame()
        mojitos = pd.DataFrame()

    if not eval_only:
        examples = pd.DataFrame()
        certas = pd.DataFrame()

        for idx in range(len(test_df)):
            rand_row = test_df.iloc[idx]
            l_id = int(rand_row['ltable_id'])
            l_tuple = lsource.iloc[l_id]
            r_id = int(rand_row['rtable_id'])
            r_tuple = rsource.iloc[r_id]

            cf_dir = exp_dir + dataset + '/' + model_name + '/' + str(idx)
            os.makedirs(cf_dir, exist_ok=True)


            prediction = get_original_prediction(l_tuple, r_tuple, predict_fn)
            class_to_explain = np.argmax(prediction)

            label = rand_row["label"]
            row_id = str(l_id) + '-' + str(r_id)
            item = get_row(l_tuple, r_tuple)

            dices = pd.DataFrame()
            limecs = pd.DataFrame()
            shapcs = pd.DataFrame()
            try:
                # CERTA
                print('certa')
                t0 = time.perf_counter()

                saliency_df, cf_summary, cf_ex, triangles, lattices = certa_explainer.explain(l_tuple, r_tuple,
                                                                                              predict_fn,
                                                                                              num_triangles=num_triangles,
                                                                                              token=token, two_step_token=False,
                                                                                              debug=False)

                latency_c = time.perf_counter() - t0


                certa_saliency = saliency_df.transpose().to_dict()[0]
                certa_row = {'summary': cf_summary, 'explanation': certa_saliency, 'type': 'certa', 'latency': latency_c,
                             'match': class_to_explain,
                             'label': label, 'row': row_id, 'prediction': prediction}

                certas = certas.append(certa_row, ignore_index=True)
                certas.to_csv(exp_dir + dataset + '/' + model_name + '/certa.csv')
                certa_dest_file = cf_dir + '/certa.csv'
                cf_ex.to_csv(certa_dest_file)

                lidx = 0
                for lattice in lattices:
                    lattice.triangle.to_csv(cf_dir + '/triangle_' + str(lidx) + '.csv')
                    try:
                        dot_lattice = lattice.hasse()
                        with open(cf_dir + '/lattice_' + str(lidx) + '.dot', 'w') as f:
                            f.write(dot_lattice)
                    except:
                        pass
                    lidx += 1

                if compare:
                    # Mojito
                    print('mojito')
                    if class_to_explain == 1:
                        t0 = time.perf_counter()
                        mojito_exp_drop = mojito.drop(predict_fn_mojito, item,
                                                      num_features=15,
                                                      num_perturbation=100)

                        latency_m = time.perf_counter() - t0
                        if token:
                            md = dict()
                            for i in range(len(mojito_exp_drop)):
                                row = mojito_exp_drop.iloc[i]
                                att = row['attribute']
                                if att not in ['id', 'rtable_id', 'ltable_id']:
                                    md[att + '__' + row['token']] = float(row['weight'])
                            mojito_exp = md
                        else:
                            mojito_exp = mojito_exp_drop.groupby('attribute')['weight'].mean().to_dict()
                    else:
                        t0 = time.perf_counter()
                        mojito_exp_copy = mojito.copy(predict_fn_mojito, item,
                                                      num_features=15,
                                                      num_perturbation=100)

                        latency_m = time.perf_counter() - t0
                        if token:
                            md = dict()
                            for i in range(len(mojito_exp_copy)):
                                row = mojito_exp_copy.iloc[i]
                                att = row['attribute']
                                if att not in ['id', 'rtable_id', 'ltable_id']:
                                    md[att + '__' + row['token']] = float(row['weight'])
                            mojito_exp = md
                        else:
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

                    if token:
                        ld = dict()
                        for i in range(len(land_explanation)):
                            row = land_explanation.iloc[i]
                            att = row['column']
                            if att not in ['id', 'rtable_id', 'ltable_id']:
                                ld[att + '__' + row['word']] = float(row['impact'])

                        land_exp = ld
                    else:
                        land_exp = land_explanation.groupby('column')['impact'].sum().to_dict()

                    land_row = {'explanation': str(land_exp), 'type': 'landmark', 'latency': latency_l,
                                'match': class_to_explain,
                                'label': label, 'row': row_id, 'prediction': prediction}
                    landmarks = landmarks.append(land_row, ignore_index=True)

                    if not token:
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

                    instance = pd.DataFrame(rand_row).transpose().astype(str)
                    for c in ['label', 'ltable_id', 'rtable_id']:
                        if c in instance.columns:
                            instance = instance.drop([c], axis=1)

                    if not os.path.exists(cf_dir + '/limec.csv'):
                        if token:
                            continue
                        print('lime-c')
                        try:
                            t0 = time.perf_counter()
                            if token:
                                limec_explainer = LimeCounterfactual(model, predict_fn_c, None, 0.5,
                                                                     to_token_df(instance), time_maximum=300,
                                                                     class_names=['nomatch_score', 'match_score'],
                                                                     token=token)
                                limec_exp = limec_explainer.explanation(to_token_df(instance))
                            else:
                                limec_exp = limec_explainer.explanation(instance)
                            print(limec_exp)
                            latency_lc = time.perf_counter() - t0
                            limec_row = {'latency': latency_lc}
                            limecs = limecs.append(limec_row, ignore_index=True)
                            if limec_exp is not None:
                                limec_exp['cf_example'].to_csv(cf_dir + '/limec.csv')
                        except:
                            print(traceback.format_exc())
                            print(f'skipped item {str(idx)}')
                            pass

                    if not os.path.exists(cf_dir + '/shapc.csv'):
                        if token:
                            continue
                        print('shap-c')
                        try:
                            t0 = time.perf_counter()
                            if token:
                                shapc_explainer = ShapCounterfactual(predict_fn_c, 0.5, to_token_df(instance),
                                                                     time_maximum=300)
                                sc_exp = shapc_explainer.explanation(to_token_df(instance), train_noids.apply(lambda x: ' '.join(x), axis = 1))
                            else:
                                sc_exp = shapc_explainer.explanation(instance, train_noids[:50])
                            latency_sc = time.perf_counter() - t0
                            shapc_row = {'latency': latency_sc}
                            shapcs = limecs.append(shapc_row, ignore_index=True)
                            print(f'{idx}- shap-c:{sc_exp}')
                            if sc_exp is not None:
                                sc_exp['cf_example'].to_csv(cf_dir + '/shapc.csv')
                        except:
                            print(traceback.format_exc())
                            print(f'skipped item {str(idx)}')
                            pass

                    if not os.path.exists(cf_dir + '/dice_random.csv'):
                        print('dice_r')
                        if token:
                            continue
                        try:
                            t0 = time.perf_counter()
                            dice_exp = exp.generate_counterfactuals(instance,
                                                                    total_CFs=10, desired_class="opposite")
                            latency_dc = time.perf_counter() - t0
                            dice_row = {'latency': latency_dc}
                            dices = dices.append(dice_row, ignore_index=True)
                            dice_exp_df = dice_exp.cf_examples_list[0].final_cfs_df
                            print(f'dice_r:{idx}:{dice_exp_df}')
                            if dice_exp_df is not None:
                                dice_exp_df.to_csv(cf_dir + '/dice_random.csv')
                        except:
                            print(traceback.format_exc())
                            print(f'skipped item {str(idx)}')
                            pass

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
        print('--latency--')
        if compare:
            mojitos.to_csv(exp_dir + dataset + '/' + model_name + '/mojito.csv')
            landmarks.to_csv(exp_dir + dataset + '/' + model_name + '/landmark.csv')
            shaps.to_csv(exp_dir + dataset + '/' + model_name + '/shap.csv')
            shapcs.to_csv(exp_dir + dataset + '/' + model_name + '/shapc.csv')
            limecs.to_csv(exp_dir + dataset + '/' + model_name + '/limec.csv')
            dices.to_csv(exp_dir + dataset + '/' + model_name + '/dice.csv')
            saliency_names = ['certa', 'landmark', 'mojito', 'shap']
            print(f"mojito: {mojitos['latency'].mean()}")
            print(f"landmark: {landmarks['latency'].mean()}")
            try:
                print(f"shap: {shaps['latency'].mean()}")
            except:
                pass
            try:
                print(f"shap-c: {shapcs['latency'].mean()}")
            except:
                pass
            try:
                print(f"lime-c: {limecs['latency'].mean()}")
            except:
                pass
            try:
                print(f"dice: {dices['latency'].mean()}")
            except:
                pass
        else:
            saliency_names = ['certa']
        examples.to_csv(exp_dir + dataset + '/' + model_name + '/examples.csv')
        certas.to_csv(exp_dir + dataset + '/' + model_name + '/certa.csv')
        print(f"certa: {certas['latency'].mean()}")
    else:
        examples = pd.read_csv(exp_dir + dataset + '/' + model_name + '/examples.csv')
        saliency_names = ['certa', 'landmark', 'mojito', 'shap']
    print('evaluating saliencies')
    try:
        faithfulness = get_faithfulness(saliency_names, model, '%s%s%s/%s' % ('', exp_dir, dataset, mtype), test_df)
        print(f'{mtype}: faithfulness for {dataset}: {faithfulness}')
        ci = get_confidence(saliency_names, exp_dir + dataset + '/' + mtype)
        print(f'{mtype}: confidence indication for {dataset}: {ci}')
        pd.DataFrame(faithfulness, index=[0]).to_csv(f'eval_faithfulness_{dataset}_{mtype}.csv')
        pd.DataFrame(ci, index=[0]).to_csv(f'eval_ci_{dataset}_{mtype}.csv')
    except:
        pass

    print('evaluating cfs')
    t = 10
    cf_eval = dict()
    cf_names = ['certa', 'dice_random', 'shapc', 'limec']
    for cf_name in cf_names:
        print(f'processing {cf_name}')
        validity = 0
        proximity = 0
        sparsity = 0
        diversity = 0
        length = 0
        count = 1e-10
        for i in range(len(examples)):
            try:
                # get cfs
                expl_df = pd.read_csv(exp_dir + dataset + '/' + model_name + '/' + str(i) + '/' + cf_name + '.csv')

                example_row = examples.iloc[i]
                instance = example_row.drop(['ltable_id', 'rtable_id', 'match', 'label'])
                score = example_row['match']
                predicted_class = int(float(score) > 0.5)

                # validity
                validity += get_validity(model, expl_df[:t], predicted_class)

                # proximity
                proximity += get_proximity(expl_df[:t], instance)

                # sparsity
                sparsity += get_sparsity(expl_df[:t], instance)

                # diversity
                diversity += get_diversity(expl_df[:t])

                length += len(expl_df)
                count += 1
            except:
                pass
        mean_validity = validity / count
        mean_proximity = proximity / count
        mean_sparsity = sparsity / count
        mean_diversity = diversity / count
        mean_length = length / count
        row = {'validity': mean_validity, 'proximity': mean_proximity,
               'sparsity': mean_sparsity, 'diversity': mean_diversity,
               'length': mean_length}
        print(f'{cf_name}:{row}')
        cf_eval[cf_name] = row
    print(f'{mtype}: cf-eval for {dataset}: {cf_eval}')
    try:
        pd.DataFrame(cf_eval, colums=['metric','certa','dice_random','shapc','limec']).to_csv(f'eval_cf_{dataset}_{mtype}.csv')
    except:
        pass


def evaluate(mtype: str, samples: int = -1, filtered_datasets: list = [], exp_dir: str = experiments_dir,
             compare=False, da=None, num_triangles=10, token=False, eval_only=False):
    if not exp_dir.endswith('/'):
        exp_dir = exp_dir + '/'
    exp_dir = exp_dir + 'all/'
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

        def predict_fn_c(x, **kwargs):
            return model.predict(x, **kwargs)['match_score']

        def predict_fn_t(x, **kwargs):
            sd = dict()
            for c in x.columns:
                at = c.split('__')
                att = at[0]
                tok = str(at[1])
                old_val = ''
                if att in sd:
                    old_val = sd.get(att)
                sd[att] = old_val + ' ' + tok
            return model.predict(pd.DataFrame.from_dict(sd, orient='index').T, **kwargs)['match_score']

        test = pd.read_csv(datadir + '/test.csv')
        lsource = pd.read_csv(datadir + '/tableA.csv')
        rsource = pd.read_csv(datadir + '/tableB.csv')
        gt = pd.read_csv(datadir + '/train.csv')
        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])[:samples]
        train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])

        eval_all(compare, dataset, exp_dir, lsource, model, model_name, mtype, predict_fn, predict_fn_mojito,
                          rsource, test_df, train_df, da, num_triangles, token, eval_only, predict_fn_c, predict_fn_t)


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
    parser.add_argument('--num_triangles', metavar='t', type=int, default=10,
                        help='no. of open triangles used to generate CERTA explanations')
    parser.add_argument('--token', metavar='tk', type=bool, default=False,
                        help='whether generating token-level explanations')
    parser.add_argument('--eval_only', metavar='eo', type=bool, default=False,
                        help='whether to regenerate evaluations on previously generated explanations')

    args = parser.parse_args()
    base_datadir = args.base_dir
    if not base_datadir.endswith('/'):
        base_datadir = base_datadir + '/'
    filtered_datasets = args.datasets
    mtype = args.model_type
    samples = args.samples
    compare = args.compare
    da = args.da
    num_triangles = args.num_triangles
    token = args.token
    eval_only = args.eval_only

    evaluate(mtype, filtered_datasets=filtered_datasets, samples=samples, compare=compare, da=da,
             num_triangles=num_triangles, token=token, eval_only=eval_only)
