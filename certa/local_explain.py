import logging
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from certa.utils import diff


def get_original_prediction(r1, r2, predict_fn):
    r1r2 = get_row(r1, r2)
    return predict_fn(r1r2)[['nomatch_score', 'match_score']].values[0]


def get_row(r1, r2, lprefix='ltable_', rprefix='rtable_'):
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    return r1r2


def support_predictions(r1: pd.Series, r2: pd.Series, lsource: pd.DataFrame,
                        rsource: pd.DataFrame, predict_fn, lprefix, rprefix, num_triangles: int = 100,
                        class_to_explain: int = None, max_predict: int = -1,
                        use_w: bool = True, use_q: bool = True, use_all: bool = False):
    '''
    generate a pd.DataFrame of support predictions to be used to generate open triangles.
    :param r1: the "left" record
    :param r2: the "right" record
    :param lsource: the "left" data source
    :param rsource: the "right" data source
    :param predict_fn: the ER model prediction function
    :param lprefix: the prefix of attributes from the "left" table
    :param rprefix: the prefix of attributes from the "right" table
    :param num_triangles: number of open triangles to be used to generate the explanation
    :param class_to_explain: the class to be explained
    :param max_predict: the maximum number of predictions to be performed by the ER model to generate the requested
        number of open triangles
    :param use_w: whether to use left open triangles
    :param use_q: whether to use right open triangles
    :param use_all: whether to use all possible records in the existing data sources to create support records, not
        stopping when _num_triangles_ records have been found
    :return: a pd.DataFrame of record pairs with one record from the original prediction and one record yielding an
        opposite prediction by the ER model
    '''
    r1r2 = get_row(r1, r2)
    original_prediction = predict_fn(r1r2)[['nomatch_score', 'match_score']].values[0]

    r1r2['id'] = "0@" + str(r1r2[lprefix + 'id'].values[0]) + "#" + "1@" + str(r1r2[rprefix + 'id'].values[0])

    copies, copies_left, copies_right = expand_copies(lprefix, lsource, r1, r2, rprefix, rsource)

    find_positives, support = get_support(class_to_explain, pd.concat([lsource, copies_left]), max_predict,
                                         original_prediction, predict_fn, r1, r2, pd.concat([rsource, copies_right]),
                                         use_w, use_q, lprefix, rprefix, num_triangles)

    if len(support) > 0:
        if len(support) > num_triangles:
            support = support.sample(n=num_triangles)
        else:
            logging.warning(f'could find {str(len(support))} triangles of the {str(num_triangles)} requested')

        support['label'] = list(map(lambda predictions: int(round(predictions)),
                                         support.match_score.values))
        support = support.drop(['match_score', 'nomatch_score'], axis=1)
        if class_to_explain == None:
            r1r2['label'] = np.argmax(original_prediction)
        else:
            r1r2['label'] = class_to_explain
        support_pairs = pd.concat([r1r2, support], ignore_index=True)
        return support_pairs, copies_left, copies_right
    else:
        logging.warning('no triangles found')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def find_candidates_predict(record, source, find_positives, predict_fn, num_candidates, lj=True,
                            max=-1, lprefix='ltable_', rprefix='rtable_', batched: bool = True):
    if lj:
        records = pd.DataFrame()
        records = records.append([record] * len(source), ignore_index=True)
        copy = source.copy()
        records.columns = list(map(lambda col: lprefix + col, records.columns))
        copy.columns = list(map(lambda col: rprefix + col, copy.columns))
        records.index = copy.index
        samples = pd.concat([records, copy], axis=1)
    else:
        copy = source.copy()
        records = pd.DataFrame()
        records = records.append([record] * len(source), ignore_index=True)
        records.index = copy.index
        copy.columns = list(map(lambda col: lprefix + col, copy.columns))
        records.columns = list(map(lambda col: rprefix + col, records.columns))
        samples = pd.concat([copy, records], axis=1)

    if max > 0:
        samples = samples.sample(frac=1)[:max]

    record2text = " ".join([str(val) for k, val in record.to_dict().items() if k not in ['id']])
    samples['score'] = samples.T.apply(lambda row: cs(record2text, " ".join(row.astype(str))))
    samples = samples.sort_values(by='score', ascending=not find_positives)
    samples = samples.drop(['score'], axis=1)
    result = pd.DataFrame()
    batch = num_candidates * 4
    splits = min(10, int(len(samples) / batch))
    i = 0
    if batched:
        while len(result) < num_candidates and i < splits:
            batch_samples = samples[batch * i:batch * (i + 1)]
            predicted = predict_fn(batch_samples)
            if find_positives:
                out = predicted[predicted["match_score"] > 0.5]
            else:
                out = predicted[predicted["match_score"] < 0.5]
            if len(out) > 0:
                result = pd.concat([result, out], axis=0)
            logging.info(f'{i}:{len(out)},{len(result)}')
            i += 1
    else:
        predicted = predict_fn(samples)
        if find_positives:
            out = predicted[predicted["match_score"] > 0.5]
        else:
            out = predicted[predicted["match_score"] < 0.5]
        if len(out) > 0:
            result = pd.concat([result, out], axis=0)
    return result


def generate_subsequences(lsource, rsource, max=-1):
    new_records_left_df = pd.DataFrame()
    for i in np.arange(len(lsource[:max])):
        r = lsource.iloc[i]
        nr_df = pd.DataFrame(generate_modified(r, start_id=len(new_records_left_df) + len(lsource)))
        if len(nr_df) > 0:
            nr_df.columns = lsource.columns
            new_records_left_df = pd.concat([new_records_left_df, nr_df])
    new_records_right_df = pd.DataFrame()
    for i in np.arange(len(rsource[:max])):
        r = rsource.iloc[i]
        nr_df = pd.DataFrame(generate_modified(r, start_id=len(new_records_right_df) + len(rsource)))
        if len(nr_df) > 0:
            nr_df.columns = rsource.columns
            new_records_right_df = pd.concat([new_records_right_df, nr_df])
    return new_records_left_df, new_records_right_df


def get_support(class_to_explain, lsource, max_predict, original_prediction, predict_fn, r1, r2,
                rsource, use_w, use_q, lprefix, rprefix, num_triangles, use_all: bool = False):
    candidates4r1 = pd.DataFrame()
    candidates4r2 = pd.DataFrame()
    num_candidates = int(num_triangles / 2)
    if class_to_explain == None:
        findPositives = bool(original_prediction[0] > original_prediction[1])
    else:
        findPositives = bool(0 == int(class_to_explain))
    if use_q:
        candidates4r1 = find_candidates_predict(r1, rsource, findPositives, predict_fn, num_candidates, batched=not use_all,
                                                lj=True, max=max_predict, lprefix=lprefix, rprefix=rprefix)
    if use_w:
        candidates4r2 = find_candidates_predict(r2, lsource, findPositives, predict_fn, num_candidates,
                                                lj=False, max=max_predict, lprefix=lprefix, rprefix=rprefix)

    # balance
    max_len = min(len(candidates4r1), len(candidates4r2))
    candidates4r1 = candidates4r1.sample(n=max_len)
    candidates4r2 = candidates4r2.sample(n=max_len)
    candidates = pd.concat([candidates4r1, candidates4r2]).sample(frac=1)

    neighborhood = pd.DataFrame()
    if len(candidates) > 0:
        candidates['id'] = "0@" + candidates[lprefix + 'id'].astype(str) + "#" + "1@" + candidates[
            rprefix + 'id'].astype(str)
        if findPositives:
            neighborhood = candidates[candidates.match_score >= 0.5].copy()
        else:
            neighborhood = candidates[candidates.match_score < 0.5].copy()

    return findPositives, neighborhood


def generate_modified(record, start_id: int = 0):
    new_copies = []
    t_len = len(record)
    copy = record.copy()
    for t in range(t_len):
        attr_value = str(copy.get(t))
        values = attr_value.split()
        for cut in range(1, len(values)):
            for new_val in [" ".join(values[cut:]),
                            " ".join(values[:cut])]:  # generate new values with prefix / suffix dropped
                new_copy = record.copy()
                new_copy[t] = new_val  # substitute the new value with missing prefix / suffix on the target attribute
                if start_id > 0:
                    new_copy['id'] = len(new_copies) + start_id
                new_copies.append(new_copy)
    return new_copies


WORD = re.compile(r'\w+')


def cs(text1, text2):
    vec1 = Counter(WORD.findall(text1))
    vec2 = Counter(WORD.findall(text2))
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def expand_copies(lprefix, lsource, r1, r2, rprefix, rsource):
    generated_df = pd.DataFrame()
    new_copies_left = []
    new_copies_right = []
    left = True
    for record in [r1, r2]:
        r1_df = pd.DataFrame(data=[record.values], columns=record.index)
        r2_df = pd.DataFrame(data=[record.values], columns=record.index)
        r1_df.columns = list(map(lambda col: 'ltable_' + col, r1_df.columns))
        r2_df.columns = list(map(lambda col: 'rtable_' + col, r2_df.columns))
        r1r2c = pd.concat([r1_df, r2_df], axis=1)

        original = r1r2c.iloc[0].copy()
        t_len = int(len(r1r2c.columns) / 2)
        # only used for reporting
        r1r2c['diff'] = ''
        r1r2c['attr_name'] = ''
        r1r2c['attr_pos'] = ''
        copy = original.copy()
        for t in range(t_len):
            if left:
                t = t_len + t
            attr_value = str(copy.get(t))
            values = attr_value.split()
            for cut in range(1, len(values)):
                for new_val in [" ".join(values[cut:]),
                                " ".join(values[:cut])]:  # generate new values with prefix / suffix dropped
                    new_copy = original.copy()
                    new_copy[
                        t] = new_val  # substitute the new value with missing prefix / suffix on the target attribute
                    if left:
                        prefix = rprefix
                        new_id = len(new_copies_left) + len(rsource)
                        idn = 'rtable_id'
                    else:
                        prefix = lprefix
                        idn = 'ltable_id'
                        new_id = len(new_copies_right) + len(lsource)

                    new_record = pd.DataFrame(new_copy).transpose().filter(regex='^' + prefix).iloc[0]
                    new_record[idn] = new_id
                    new_copy[idn] = new_id
                    if left:
                        new_copies_left.append(new_record)
                    else:
                        new_copies_right.append(new_record)

                    # only used for reporting
                    new_copy['diff'] = diff(attr_value, new_val)
                    new_copy['attr_name'] = r1r2c.columns[t]
                    new_copy['attr_pos'] = t

                    r1r2c = r1r2c.append(new_copy, ignore_index=True)
        if left:
            r1r2c['id'] = "0@" + r1r2c[lprefix + 'id'].astype(str) + "#" + "1@" + r1r2c[
                rprefix + 'id'].astype(str)
            left = False
        else:
            r1r2c['id'] = "0@" + r1r2c[lprefix + 'id'].astype(str) + "#" + "1@" + r1r2c[
                rprefix + 'id'].astype(str)

        generated_df = pd.concat([generated_df, r1r2c], axis=0)
    generated_records_left_df = pd.DataFrame(new_copies_left).rename(columns=lambda x: x[len(lprefix):])
    generated_records_right_df = pd.DataFrame(new_copies_right).rename(columns=lambda x: x[len(rprefix):])

    return generated_df, generated_records_left_df, generated_records_right_df


def get_neighbors(find_positives, predict_fn, r1r2c, report: bool = False):
    original = r1r2c.copy()
    try:
        r1r2c = r1r2c.drop(columns=['diff', 'attr_name', 'attr_pos'])
    except:
        pass

    unlabeled_predictions = predict_fn(r1r2c)
    if report:
        try:
            report = pd.concat([original, unlabeled_predictions['match_score']], axis=1)
            report.to_csv('experiments/diffs.csv', mode='a')
        except:
            pass
    if find_positives:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score >= 0.5].copy()
    else:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score < 0.5].copy()
    return neighborhood


def interleave_dataframes(dfs=None):
    if dfs is None:
        dfs = [pd.DataFrame(np.zeros(10)), pd.DataFrame(np.ones(25)), pd.DataFrame(np.ones(
            8) * 2)]  # varialbe-sized pandas dataframe, these should have the same structure (i.e. # and order of the columns)

    num_classes = len(dfs)

    counters = [0 for i in
                range(len(dfs))]  # these counters are used to check when a certain dataframe has been entirely consumed
    tot_len = sum([len(x) for x in dfs])
    new_df = pd.DataFrame(columns=dfs[0].columns, index=range(
        tot_len))  # new dataframe will have length=tot_len and same structure as input dataframes

    for i in range(tot_len):
        added = False
        sel = i % num_classes
        while not added:
            try:
                new_df.ix[i] = dfs[sel].iloc[counters[sel]]  # if the dfs[sel] has still elements to process..
                counters[sel] += 1
                added = True
            except:
                sel = (sel + 1) % num_classes  # otherwise try with the next dataframe until all of them have been consumed

    return new_df