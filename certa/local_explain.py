import pandas as pd
import math, re, os, random, string
from collections import Counter
import numpy as np
from certa.edit_dna import Sequence

'''
N.B. For now this script can only work using deepmatcher
'''

WORD = re.compile(r'\w+')


# calculate similarity between two text vectors
def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def find_candidates(record, source, similarity_threshold, find_positives, lj=True):
    record2text = " ".join([str(val) for k, val in record.to_dict().items() if k not in ['id']])
    source_without_id = source.copy()
    source_without_id = source_without_id.drop(['id'], axis=1)
    source_ids = source.id.values
    # for a faster iteration
    source_without_id = source_without_id.values
    candidates = []
    for idx, row in enumerate(source_without_id):
        currentRecord = " ".join(row.astype(str))
        currentSimilarity = get_cosine(record2text, currentRecord)
        if find_positives:
            if currentSimilarity >= similarity_threshold:
                if lj:
                    candidates.append((record['id'], source_ids[idx]))
                else:
                    candidates.append((source_ids[idx], record['id']))
        else:
            if currentSimilarity < similarity_threshold:
                if lj:
                    candidates.append((record['id'], source_ids[idx]))
                else:
                    candidates.append((source_ids[idx], record['id']))
    return pd.DataFrame(candidates, columns=['ltable_id', 'rtable_id'])


def get_original_prediction(r1, r2, predict_fn):
    lprefix = 'ltable_'
    rprefix = 'rtable_'
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    #r1r2['id'] = "0@" + str(r1r2[lprefix + 'id'].values[0]) + "#" + "1@" + str(r1r2[rprefix + 'id'].values[0])
    #r1r2 = r1r2.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    return predict_fn(r1r2, None)[['nomatch_score', 'match_score']].values[0]


def find_candidates_predict(record, source, similarity_threshold, find_positives, predict_fn, lj=True):
    source_without_id = source.copy()
    source_without_id = source_without_id.drop(['id'], axis=1)
    record_without_id = record.copy().drop(['id'])
    source_ids = source.id.values
    # for a faster iteration
    #source_without_id = source_without_id.values
    candidates = []
    for idx in range(len(source_without_id)):
        row = source_without_id.iloc[idx]
        prediction = get_original_prediction(record_without_id, row, predict_fn)
        if find_positives:
            if prediction[1] >= similarity_threshold:
                if lj:
                    candidates.append((record['id'], source_ids[idx]))
                else:
                    candidates.append((source_ids[idx], record['id']))
        else:
            if prediction[1] <= similarity_threshold:
                if lj:
                    candidates.append((record['id'], source_ids[idx]))
                else:
                    candidates.append((source_ids[idx], record['id']))
    return pd.DataFrame(candidates, columns=['ltable_id', 'rtable_id'])


def __generate_unlabeled(dataset_dir, unlabeled_filename, lprefix='ltable_', rprefix='rtable_'):
    df_tableA = pd.read_csv(os.path.join(dataset_dir, 'tableA.csv'), dtype=str)
    df_tableB = pd.read_csv(os.path.join(dataset_dir, 'tableB.csv'), dtype=str)
    unlabeled_ids = pd.read_csv(os.path.join(dataset_dir, unlabeled_filename), dtype=str)
    unlabeled_ids.columns = ['id1', 'id2']
    left_columns = list(map(lambda s: lprefix + s, list(df_tableA)))
    right_columns = list(map(lambda s: rprefix + s, list(df_tableB)))
    df_tableA.columns = left_columns
    df_tableB.columns = right_columns

    unlabeled_df = unlabeled_ids.merge(df_tableA, how='inner', left_on='id1', right_on=lprefix + 'id') \
        .merge(df_tableB, how='inner', left_on='id2', right_on=rprefix + 'id')
    unlabeled_df[lprefix + 'id'] = unlabeled_df[lprefix + 'id'].astype(str)
    unlabeled_df[rprefix + 'id'] = unlabeled_df[rprefix + 'id'].astype(str)
    unlabeled_df['id'] = "0@" + unlabeled_df[lprefix + 'id'] + "#" + "1@" + unlabeled_df[rprefix + 'id']
    unlabeled_df = unlabeled_df.drop(['id1', 'id2', lprefix + 'id', rprefix + 'id'], axis=1)
    return unlabeled_df.drop_duplicates()


def copy_EDIT(series, n, d):
    copy = series.copy()
    if n == -1:
        l_idx = random.randint(0, int(len(series)/2))
        r_idx = l_idx + int(len(series)/2)
        o_l_val = str(copy.get(l_idx))
        l_val = o_l_val
        while l_val == o_l_val:
            l_val = copy_EDIT_match([o_l_val], d)[0]
        o_r_val = str(copy.get(r_idx))
        r_val = o_r_val
        while o_r_val == r_val:
            r_val = copy_EDIT_match([o_r_val], d)[0]
        copy.update(pd.Series([l_val, r_val], index=[l_idx, r_idx]))
    else:
        o_r_val = str(copy.get(n))
        r_val = o_r_val
        while o_r_val == r_val:
            changed = copy_EDIT_match([o_r_val], d)
            r_val = changed[0]
        copy[n] = r_val
    return copy


def copy_EDIT_match(tupla, d):
    copy_tup = []

    for i in range(len(tupla)):
        attr = Sequence(tupla[i])
        if len(tupla[i]) > 1:
            n = 3  # number of strings in result
            mutates = attr.mutate(d, n)
            copy_tup.append(str(mutates[1]))
        else:
            copy_tup.append(tupla[i])

    if copy_tup == tupla:
        copy_tup = copy_EDIT_match(tupla, d)
    # print(copy_tup)
    return copy_tup


def dataset_local(r1: pd.Series, r2: pd.Series, model, lsource: pd.DataFrame,
                  rsource: pd.DataFrame, dataset_dir, theta_min: float,
                  theta_max: float, predict_fn, num_triangles: int = 100, class_to_explain: int = None,
                  use_predict: bool = True, generate_perturb: bool = False):
    lprefix = 'ltable_'
    rprefix = 'rtable_'
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    r1r2['id'] = "0@" + str(r1r2[lprefix + 'id'].values[0]) + "#" + "1@" + str(r1r2[rprefix + 'id'].values[0])
    r1r2 = r1r2.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    originalPrediction = predict_fn(r1r2, model)[['nomatch_score', 'match_score']].values[0]

    if class_to_explain == None:
       findPositives = bool(originalPrediction[0] > originalPrediction[1])
    else:
        findPositives = bool(0 == int(class_to_explain))
    if findPositives:
        if use_predict:
            candidates4r1 = find_candidates_predict(r1, rsource, theta_max, findPositives, predict_fn, lj=True)
            candidates4r2 = find_candidates_predict(r2, lsource, theta_max, findPositives, predict_fn, lj=False)
        else:
            candidates4r1 = find_candidates(r1, rsource, theta_max, find_positives=findPositives, lj=True)
            candidates4r2 = find_candidates(r2, lsource, theta_max, find_positives=findPositives, lj=False)
    else:
        if use_predict:
            candidates4r1 = find_candidates_predict(r1, rsource, theta_min, findPositives, predict_fn, lj=True)
            candidates4r2 = find_candidates_predict(r2, lsource, theta_min, findPositives, predict_fn, lj=False)
        else:
            candidates4r1 = find_candidates(r1, rsource, theta_min, find_positives=findPositives, lj=True)
            candidates4r2 = find_candidates(r2, lsource, theta_min, find_positives=findPositives, lj=False)
    id4explanation = pd.concat([candidates4r1, candidates4r2], ignore_index=True)
    tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
    id4explanation.to_csv(os.path.join(dataset_dir, tmp_name), index=False)
    unlabeled_df = __generate_unlabeled(dataset_dir, tmp_name)
    os.remove(os.path.join(dataset_dir, tmp_name))
    neighborhood = pd.DataFrame()
    if len(unlabeled_df) > 0:
        neighborhood = get_neighbors(findPositives, model, predict_fn, unlabeled_df)
        if len(neighborhood) > num_triangles:
            neighborhood = neighborhood.sample(n=num_triangles)
        else:
            print(f'could find {len(neighborhood)} neighbors of the {num_triangles} requested')

    if generate_perturb and len(neighborhood) < num_triangles:
        r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
        r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
        r1_df.columns = list(map(lambda col: 'ltable_' + col, r1_df.columns))
        r2_df.columns = list(map(lambda col: 'rtable_' + col, r2_df.columns))
        r1r2c = pd.concat([r1_df, r2_df], axis=1)
        original = r1r2c.iloc[0].copy()
        for i in range(1, num_triangles):
            t_len = len(r1r2c.columns)
            for n in range(int(t_len / 2) + 1, t_len):
                copy = original.copy()
                mp = len(str(copy.get(n)))
                for t in range(1, mp - 1):
                    edit = copy_EDIT(copy, n, t)
                    r1r2c = r1r2c.append(edit, ignore_index=True)
        r1r2c['id'] = "0@" + r1r2c[lprefix+'id'].astype(str) + "#" + "1@" + r1r2c[rprefix+'id'].astype(str)
        neighborhood = pd.concat([neighborhood, get_neighbors(findPositives, model, predict_fn, r1r2c)], axis=0)
        print(f'copy-edit neighborhood: {len(neighborhood)}')

    if len(neighborhood) > 0:
        neighborhood['label'] = list(map(lambda predictions: int(round(predictions)),
                                         neighborhood.match_score.values))
        neighborhood = neighborhood.drop(['match_score', 'nomatch_score'], axis=1)
        if class_to_explain == None:
            r1r2['label'] = np.argmax(originalPrediction)
        else:
            r1r2['label'] = class_to_explain
        dataset4explanation = pd.concat([r1r2, neighborhood], ignore_index=True)
        return dataset4explanation
    else:
        return pd.DataFrame()


def get_neighbors(findPositives, model, predict_fn, r1r2c):
    unlabeled_predictions = predict_fn(r1r2c, model)
    if findPositives:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score >= 0.5].copy()
    else:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score < 0.5].copy()
    return neighborhood


def find_thresholds(test_df: pd.DataFrame, m: float):
    lprefix = 'ltable_'
    rprefix = 'rtable_'
    ignore_columns = ['id']

    l_columns = [col for col in list(test_df) if (col.startswith(lprefix)) and (col not in ignore_columns)]
    r_columns = [col for col in list(test_df) if col.startswith(rprefix) and (col not in ignore_columns)]

    l_string_test_df = test_df[l_columns].astype('str').agg(' '.join, axis=1)
    r_string_test_df = test_df[r_columns].astype('str').agg(' '.join, axis=1)
    label_df = test_df['label']

    merged_string = pd.concat([l_string_test_df, r_string_test_df, label_df], ignore_index=True, axis=1)

    sim_df = merged_string.apply(lambda x: get_cosine(x[0], x[1]), axis=1)

    tuples_ls_df = pd.concat([merged_string, sim_df], ignore_index=True, axis=1)

    lpos_df = tuples_ls_df[tuples_ls_df[2] == 1]
    lneg_df = tuples_ls_df[tuples_ls_df[2] == 0]

    theta_max = lpos_df[3].mean() + m * lpos_df[3].std()
    theta_min = lneg_df[3].mean() - m * lneg_df[3].std()

    return theta_min, theta_max
