import os
from certa.utils import merge_sources
from certa.models.utils import from_type

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback
import random
from certa.local_explain import get_row

root_datadir = '../cheapER/datasets/'
experiments_dir = 'experiments/'
d_dict = {'abt_buy':'AB', 'beers': 'BA', 'itunes_amazon':'IA'}
dataset = 'abt_buy'
target_dir = d_dict[dataset]
datadir = os.path.join(root_datadir, dataset)
mtype = 'dm'
mdir_dict = {'ditto':'Ditto', 'deeper': 'DeepER', 'dm':'DeepMatcher'}
model_dir = mdir_dict[mtype]

model = from_type(mtype)

save_path = 'models/saved/'+mtype+'/' + dataset
model.load(save_path)

def predict_fn(x):
    return model.predict(x)

def alter_data(explanations, l_tuple, r_tuple, predict_fn, agg=True, lprefix='ltable_', rprefix='rtable_',
               num_masked: int = 3, perturb: str = 'mask', mask_token: str = ''):
    data = pd.DataFrame()
    lt = l_tuple.copy()
    rt = r_tuple.copy()
    row = get_row(lt, rt)
    orig = predict_fn(row)[['nomatch_score', 'match_score']].values[0][1]
    margin = 0.5
    for explanation in explanations:
        saliency = explanation.copy()
        exp_type = saliency.pop('type')
        scores_d = []
        # scores_c = []
        for tk in np.arange(num_masked):
            # get top k important attributes
            if not agg and tk >= len(saliency):
                break
            if exp_type == 'certa':
                if agg:
                    explanation_attributes = sorted(saliency, key=saliency.get, reverse=True)[:tk]
                else:
                    explanation_attributes = [sorted(saliency, key=saliency.get, reverse=True)[tk]]
            elif orig < 0.5:
                saliency = {k: v for k, v in saliency.items()}
                if agg:
                    explanation_attributes = sorted(saliency, key=saliency.get)[:tk]
                else:
                    explanation_attributes = [sorted(saliency, key=saliency.get)[tk]]
            else:
                saliency = {k: v for k, v in saliency.items()}
                if agg:
                    explanation_attributes = sorted(saliency, key=saliency.get, reverse=True)[:tk]
                else:
                    explanation_attributes = [sorted(saliency, key=saliency.get, reverse=True)[tk]]
            # alter those attributes
            if len(explanation_attributes) > 0:
                if isinstance(perturb, pd.Series):
                    try:
                        random_record = perturb
                        lt = l_tuple.copy()
                        rt = r_tuple.copy()
                        modified_row = get_row(lt, rt)
                        for e in explanation_attributes:
                            modified_row[e] = random_record[e]
                        modified_tuple_prediction = predict_fn(modified_row)[['nomatch_score', 'match_score']].values[0]
                        score_drop = modified_tuple_prediction[1]
                        scores_d.append(score_drop)
                    except Exception as e:
                        print(traceback.format_exc())
                elif perturb == 'mask':
                    try:
                        lt = l_tuple.copy()
                        rt = r_tuple.copy()
                        modified_row = get_row(lt, rt)
                        for e in explanation_attributes:
                            modified_row[e] = mask_token
                        modified_tuple_prediction = predict_fn(modified_row)[['nomatch_score', 'match_score']].values[0]
                        score_drop = modified_tuple_prediction[1]
                        scores_d.append(score_drop)
                    except Exception as e:
                        print(traceback.format_exc())
                elif perturb == 'copy':
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
                        scores_d.append(score_copy)
                    except Exception as e:
                        print(traceback.format_exc())
        data[exp_type] = pd.Series(scores_d)
    print(scores_d)
    data['prediction'] = orig
    data['margin'] = margin
    return data


def saliency_graph(saliency_df: pd.DataFrame, l_tuple, r_tuple, predict_fn, etype='certa', nm=9, color='skyblue',
                   so=False, mapping_dict=None, perturb='mask', path=None, pred=None):
    print(pred)
    print(perturb)
    saliency = saliency_df.copy()
    if 'ltable_id' in saliency.columns:
        saliency = saliency.drop('ltable_id', axis=1)
    if 'rtable_id' in saliency.columns:
        saliency = saliency.drop('rtable_id', axis=1)
    exp = saliency.to_dict(orient='list')
    exp['type'] = etype
    if perturb == 'rand':
        perturb = train_df.iloc[random.randint(0, len(train_df) - 1)]
    single = alter_data([exp], l_tuple, r_tuple, predict_fn, num_masked=nm, agg=False, perturb=perturb)[etype]
    aggr = alter_data([exp], l_tuple, r_tuple, predict_fn, num_masked=nm + 1, perturb=perturb)[etype]
    # rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    # my_cmap = plt.get_cmap("Blues")
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=60, fontsize=15)
    if mapping_dict is not None:
        saliency_df = saliency_df.rename(columns=mapping_dict)
    if pred is not None:
        so = etype != 'certa' and float(pred) < 0.5
    saliency_sorted_df = saliency_df.sort_values(saliency_df.last_valid_index(), axis=1, ascending=so)

    x = saliency_sorted_df.columns[:nm]
    y = saliency_sorted_df.values[0][:nm]

    axes2 = plt.twinx()
    plt.bar(x=x, height=y, label='Saliency', color=color)

    y1 = aggr
    y2 = single
    ml = np.empty(len(y1))
    ml.fill(0.5)

    x1 = x[:len(y1)]
    x2 = x[:len(y2)]
    axes2.plot(x1, y1[:len(x1)], '^g-', label='Aggregate', linewidth=5.0)
    axes2.plot(x2, y2[:len(x2)], '^r-', label='Single', linewidth=5.0)
    pred_line = 0
    if pred is not None:
        pred_line = np.empty(len(y1))
        pred_line.fill(pred)
        x3 = x[:len(y1)]
        axes2.plot(x3, pred_line[:len(x3)], '^b-', label='Prediction', linewidth=1.0)
        axes2.plot(x3, ml, color='black', label='Decision boundary', linewidth=1.0)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')

    if etype == 'certa':
        plt.ylim([0.0, 1.0])
    else:
        plt.ylim(min(y) * 1.1, 1.0)  # max(max(y1),max(y2),max(pred_line)))

    plt.show()
    aggr = np.array(y1)
    sing = np.array(y2)

    if len(aggr) < len(saliency_df.columns):
        pl = len(saliency_df.columns) - len(aggr)
        aggr = np.pad(aggr, (0, pl))
    if len(sing) < len(saliency_df.columns):
        pl = len(saliency_df.columns) - len(sing)
        sing = np.pad(sing, (0, pl))

    saliency_sorted_df.loc[1] = aggr
    saliency_sorted_df.loc[2] = sing
    saliency_sorted_df['result'] = ['saliency', 'aggregate', 'single']

    return saliency_sorted_df


lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')

test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])
train_df = merge_sources(gt, 'ltable_', 'rtable_', lsource, rsource, ['label'], ['id'])


explanations = dict()
explanations['certa'] = pd.read_csv('../certem/data_new/'+target_dir+'/'+model_dir+'/certa.csv')
explanations['mojito'] = pd.read_csv('../certem/data_new/'+target_dir+'/'+model_dir+'/mojito.csv')
explanations['landmark'] =  pd.read_csv('../certem/data_new/'+target_dir+'/'+model_dir+'/landmark.csv')
explanations['shap'] =  pd.read_csv('../certem/data_new/'+target_dir+'/'+model_dir+'/shap.csv')


for etype in ['certa', 'mojito', 'shap', 'landmark']:
    for i in range(10):
        rand_row =  test_df.iloc[i]
        l_id = int(rand_row['ltable_id'])
        label = rand_row["label"]
        l_tuple = lsource.iloc[l_id]
        r_id = int(rand_row['rtable_id'])
        r_tuple = rsource.iloc[r_id]
        saliency_df = pd.DataFrame(eval(explanations[etype].iloc[i]['explanation']),index=[0])
        for perturb in ['mask','copy','rand']:
            try:
                sg_dir = '../certem/data_new/' + target_dir + '/' + model_dir + '/' + str(i) + '/sg/'
                if not os.path.exists(sg_dir):
                    os.makedirs(sg_dir)
                dpath = sg_dir+etype+'_'+perturb+'.png'
                print(dpath)
                saliency_graph(saliency_df, l_tuple, r_tuple, predict_fn, nm=5, pred=float(predict_fn(get_row(l_tuple,r_tuple))['match_score'][0]),
                           perturb=perturb, path=dpath, etype=etype)
            except:
                pass
