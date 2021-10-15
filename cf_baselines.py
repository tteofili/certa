import os
import traceback
import logging

import numpy as np
import pandas as pd

from baselines.lime_c import LimeCounterfactual
from baselines.shap_c import ShapCounterfactual
from certa.utils import merge_sources
from models.utils import from_type

import dice_ml

dice_r = True
shap_c = True
lime_c = True

# dice_g = False
# dice_k = False
# proto = False
# simple = False
# sedc = False
# dataset = 'beers'
# model_type = 'dm'
# model = from_type(model_type)
# model.load('models/' + model_type + '/' + dataset)
#
#
# def predict_fn(x):
#     return model.predict_proba(x)
#
#
# datadir = 'datasets/' + dataset
# lsource = pd.read_csv(datadir + '/tableA.csv')
# rsource = pd.read_csv(datadir + '/tableB.csv')
# gt = pd.read_csv(datadir + '/train.csv')
# valid = pd.read_csv(datadir + '/valid.csv')
# test = pd.read_csv(datadir + '/test.csv')
#
# test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, [], ['label'])
# train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, [], ['label'])
# train_noids = train_df.drop(['ltable_id', 'rtable_id'], axis=1).astype(str)
#
# test_df['outcome'] = np.argmax(model.predict_proba(test_df), axis=1)
#
# basedir = 'cf/' + dataset + '/' + model_type + '/'
# if not os.path.exists(basedir):
#     os.mkdir(basedir)
#
# for idx in range(50):
#
#     item_dir = basedir + '/' + str(idx)
#     if not os.path.exists(item_dir):
#         os.mkdir(item_dir)
#
#     rand_row = test_df.iloc[idx]
#     l_id = int(rand_row['ltable_id'])
#     l_tuple = lsource.iloc[l_id]
#     r_id = int(rand_row['rtable_id'])
#     r_tuple = rsource.iloc[r_id]
#     rand_row.head()
#     instance = pd.DataFrame(rand_row).transpose().drop(['outcome', 'ltable_id', 'rtable_id'], axis=1).astype(str)
#     classifier_fn = lambda x: model.predict_proba(x, given_columns=train_df.columns)[1, :]
#     instance_text = str(instance.values)
#
#     if lime_c:
#         print('lime-c')
#         try:
#             preprocess_cf = Preprocess_LimeCounterfactual(False)
#             vectorizer, feature_names = preprocess_cf.fit_vectorizer(instance)
#             # c = make_pipeline(vectorizer, model)
#             limec_explainer = LimeCounterfactual(model, predict_fn, vectorizer, 0.5, train_noids.columns, time_maximum=300)
#             limec_exp = limec_explainer.explanation(instance)
#             print(limec_exp)
#             if limec_exp is not None:
#                 limec_exp['cf_example'].to_csv(basedir + str(idx) + '/limec.csv')
#         except:
#             print(traceback.format_exc())
#             print(f'skipped item {str(idx)}')
#             pass
#
#     if shap_c:
#         print('shap-c')
#         try:
#             shap_predict = lambda x: predict_fn(x)['match_score'].values
#             shapc_explainer = ShapCounterfactual(predict_fn, 0.5,
#                                                  train_noids.columns, time_maximum=300)
#
#             sc_exp = shapc_explainer.explanation(instance, train_noids[:50])
#             print(f'{idx}- shap-c:{sc_exp}')
#             if sc_exp is not None:
#                 sc_exp['cf_example'].to_csv(basedir + str(idx) + '/shapc.csv')
#         except:
#             print(traceback.format_exc())
#             print(f'skipped item {str(idx)}')
#             pass
#
#     if dice_g:
#         print('dice_g')
#         try:
#             d = dice_ml.Data(dataframe=test_df.drop(['ltable_id', 'rtable_id'], axis=1),
#                              continuous_features=[],
#                              outcome_name='outcome')
#             # genetic
#             m = dice_ml.Model(model=model, backend='sklearn')
#             exp = dice_ml.Dice(d, m, method='genetic')
#             dice_exp = exp.generate_counterfactuals(instance,
#                                                     total_CFs=10, desired_class="opposite")
#             dice_exp_df = dice_exp.cf_examples_list[0].final_cfs_df
#             print(f'genetic:{idx}:{dice_exp_df}')
#         except:
#             print(traceback.format_exc())
#             print(f'skipped item {str(idx)}')
#             pass
#
#     if dice_k:
#         print('dice_k')
#         try:
#
#             d = dice_ml.Data(dataframe=test_df.drop(['ltable_id', 'rtable_id'], axis=1),
#                              continuous_features=[],
#                              outcome_name='outcome')
#             # kdtree
#             m = dice_ml.Model(model=model, backend='sklearn')
#             exp = dice_ml.Dice(d, m, method='kdtree')
#             dice_exp = exp.generate_counterfactuals(instance,
#                                                     total_CFs=10, desired_class="opposite")
#             dice_exp_df = dice_exp.cf_examples_list[0].final_cfs_df
#             print(f'kdtree:{idx}:{dice_exp_df}')
#         except:
#             print(traceback.format_exc())
#             print(f'skipped item {str(idx)}')
#             pass
#
#     if dice_r:
#         print('dice_r')
#         try:
#
#             d = dice_ml.Data(dataframe=test_df.drop(['ltable_id', 'rtable_id'], axis=1),
#                              continuous_features=[],
#                              outcome_name='outcome')
#             # random
#             m = dice_ml.Model(model=model, backend='sklearn')
#             exp = dice_ml.Dice(d, m, method='random')
#             dice_exp = exp.generate_counterfactuals(instance,
#                                                     total_CFs=10, desired_class="opposite")
#             dice_exp_df = dice_exp.cf_examples_list[0].final_cfs_df
#             print(f'random:{idx}:{dice_exp_df}')
#             if dice_exp_df is not None:
#                 # dice_exp_df[dice_exp_df['outcome'] != test_df.iloc[idx]['outcome']].to_csv('cf/'+dataset+'/'+model_type+'/'+str(idx)+'/dice_random.csv')
#                 dice_exp_df.to_csv(basedir + str(idx) + '/dice_random.csv')
#         except:
#             print(traceback.format_exc())
#             print(f'skipped item {str(idx)}')
#             pass
#
#     shape = (1,) + ((len(train_df.columns) - 2),)
#
#     instance = pd.DataFrame(rand_row).transpose().drop(['ltable_id', 'rtable_id', 'outcome'], axis=1).values
#     if proto:
#         print('proto')
#         try:
#             cf_proto = CounterfactualProto(predict_fn, shape,
#                                            feature_range=(str(train_df.min(axis=0)), str(train_df.max(axis=0))))
#             cf_proto.fit(train_df.instance(['ltable_id', 'rtable_id'], axis=1).values)
#             proto_ex = cf_proto.explain(instance)
#             print(f'{proto_ex}')
#         except:
#             print(traceback.format_exc())
#             print(f'skipped item {str(idx)}')
#             pass
#
#     if simple:
#         print('simple')
#         try:
#             cf = Counterfactual(predict_fn, shape=shape,
#                                 feature_range=(str(train_df.min(axis=0)), str(train_df.max(axis=0))))
#             simple_ex = cf.explain(instance)
#             print(f'{simple_ex}')
#         except:
#             print(traceback.format_exc())
#             print(f'skipped item {str(idx)}')
#             pass

import warnings

warnings.filterwarnings("ignore")

root_datadir = 'datasets/'
experiments_dir = 'cf/'

def baselines_gen(model, samples, filtered_datasets, exp_dir: str = experiments_dir,):
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

                test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, [], ['label'])[:samples]
                train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, [], ['label'])
                train_noids = train_df.drop(['ltable_id', 'rtable_id'], axis=1).astype(str)

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

                def predict_fn(x):
                    return model.predict_proba(x)

                test_df['outcome'] = np.argmax(model.predict_proba(test_df), axis=1)

                for idx in range(len(test_df)):
                    rand_row = test_df.iloc[idx]

                    basedir = exp_dir + dir + '/' + model_name + '/' + str(idx)

                    try:
                        instance = pd.DataFrame(rand_row).transpose().drop(['outcome', 'ltable_id', 'rtable_id'],
                                                                           axis=1).astype(str)

                        if lime_c:
                            print('lime-c')
                            try:
                                limec_explainer = LimeCounterfactual(model, predict_fn, None, 0.5,
                                                                     train_noids.columns, time_maximum=300)
                                limec_exp = limec_explainer.explanation(instance)
                                print(limec_exp)
                                if limec_exp is not None:
                                    limec_exp['cf_example'].to_csv(basedir + '/limec.csv')
                            except:
                                print(traceback.format_exc())
                                print(f'skipped item {str(idx)}')
                                pass

                        if shap_c:
                            print('shap-c')
                            try:
                                shapc_explainer = ShapCounterfactual(predict_fn, 0.5,
                                                                     train_noids.columns, time_maximum=300)

                                sc_exp = shapc_explainer.explanation(instance, train_noids[:50])
                                print(f'{idx}- shap-c:{sc_exp}')
                                if sc_exp is not None:
                                    sc_exp['cf_example'].to_csv(basedir + '/shapc.csv')
                            except:
                                print(traceback.format_exc())
                                print(f'skipped item {str(idx)}')
                                pass

                        if dice_r:
                            print('dice_r')
                            try:

                                d = dice_ml.Data(dataframe=test_df.drop(['ltable_id', 'rtable_id'], axis=1),
                                                 continuous_features=[],
                                                 outcome_name='outcome')
                                # random
                                m = dice_ml.Model(model=model, backend='sklearn')
                                exp = dice_ml.Dice(d, m, method='random')
                                dice_exp = exp.generate_counterfactuals(instance,
                                                                        total_CFs=10, desired_class="opposite")
                                dice_exp_df = dice_exp.cf_examples_list[0].final_cfs_df
                                print(f'random:{idx}:{dice_exp_df}')
                                if dice_exp_df is not None:
                                    # dice_exp_df[dice_exp_df['outcome'] != test_df.iloc[idx]['outcome']].to_csv('cf/'+dataset+'/'+model_type+'/'+str(idx)+'/dice_random.csv')
                                    dice_exp_df.to_csv(basedir + '/dice_random.csv')
                            except:
                                print(traceback.format_exc())
                                print(f'skipped item {str(idx)}')
                                pass

                    except:
                        print(traceback.format_exc())
                        print(f'skipped item {str(idx)}')
                        pass


if __name__ == "__main__":
    samples = 50
    mtype = 'deeper'
    filtered_datasets = ['dirty_dblp_scholar', 'dirty_amazon_itunes', 'dirty_walmart_amazon', 'dirty_dblp_acm',
                         'abt_buy', 'fodo_zaga', 'beers',
                         'amazon_google', 'itunes_amazon', 'walmart_amazon',
                         'dblp_scholar', 'dblp_acm']
    model = from_type(mtype)
    baselines_gen(model, samples=samples, filtered_datasets=filtered_datasets)
