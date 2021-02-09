import pandas as pd
import math, re, os, random, string
from collections import Counter
import numpy as np

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


def find_candidates(record, source, similarity_threshold, find_positives):
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
                candidates.append((record['id'], source_ids[idx]))
        else:
            if currentSimilarity < similarity_threshold:
                candidates.append((record['id'], source_ids[idx]))
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


def dataset_local(r1: pd.Series, r2: pd.Series, model, lsource: pd.DataFrame,
                  rsource: pd.DataFrame, dataset_dir, theta_min: float,
                  theta_max: float, predict_fn, num_triangles=100, class_to_explain=None):
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
        candidates4r1 = find_candidates(r1, rsource, theta_max, find_positives=findPositives)
        candidates4r2 = find_candidates(r2, lsource, theta_max, find_positives=findPositives)
    else:
        candidates4r1 = find_candidates(r1, rsource, theta_min, find_positives=findPositives)
        candidates4r2 = find_candidates(r2, lsource, theta_min, find_positives=findPositives)
    id4explanation = pd.concat([candidates4r1, candidates4r2], ignore_index=True)
    tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
    id4explanation.to_csv(os.path.join(dataset_dir, tmp_name), index=False)
    unlabeled_df = __generate_unlabeled(dataset_dir, tmp_name)
    os.remove(os.path.join(dataset_dir, tmp_name))
    unlabeled_predictions = predict_fn(unlabeled_df, model)
    if findPositives:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score >= 0.5].copy()
    else:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score < 0.5].copy()
    if len(neighborhood) > num_triangles:
        neighborhood = neighborhood.sample(n=num_triangles)
    else:
        print(f'could only find {len(neighborhood)} triangles of the {num_triangles} requested')

    neighborhood['label'] = list(map(lambda predictions: int(round(predictions)),
                                     neighborhood.match_score.values))
    neighborhood = neighborhood.drop(['match_score', 'nomatch_score'], axis=1)
    r1r2['label'] = np.argmax(originalPrediction)
    dataset4explanation = pd.concat([r1r2, neighborhood], ignore_index=True)
    return dataset4explanation


def find_similarities(test_df: pd.DataFrame, strict: bool):
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

    theta_max = lpos_df[3].mean()
    theta_min = lneg_df[3].mean()

    if strict:
        theta_max = theta_max + lpos_df[3].std()
        theta_min = theta_min - lneg_df[3].std()

    return min(theta_min, theta_max), max(theta_min, theta_max)
