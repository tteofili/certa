import logging
from collections import defaultdict
from functools import partialmethod
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def _renameColumnsWithPrefix(prefix, df):
    newcol = []
    for col in list(df):
        newcol.append(prefix + col)
    df.columns = newcol


def _powerset(xs, minlen, maxlen):
    return [subset for i in range(minlen, maxlen + 1)
            for subset in combinations(xs, i)]


def getMixedTriangles(dataset, sources):
    # a triangle is a triple <u, v, w> where <u, v> is a match and <v, w> is a non-match (<u,w> should be a non-match)
    triangles = []
    # to not alter original dataset
    dataset_c = dataset.copy()
    sourcesmap = {}
    # the id is so composed: lsourcenumber@id#rsourcenumber@id
    for i in range(len(sources)):
        sourcesmap[i] = sources[i]
    dataset_c['ltable_id'] = list(map(lambda lrid: str(lrid).split("#")[0], dataset_c.id.values))
    dataset_c['rtable_id'] = list(map(lambda lrid: str(lrid).split("#")[1], dataset_c.id.values))
    positives = dataset_c[dataset_c.label == 1].astype('str')  # match classified samples
    negatives = dataset_c[dataset_c.label == 0].astype('str')  # no-match classified samples
    l_pos_ids = positives.ltable_id.astype('str').values  # left ids of positive samples
    r_pos_ids = positives.rtable_id.astype('str').values  # right ids of positive samples
    for lid, rid in zip(l_pos_ids, r_pos_ids):  # iterate through positive l_id, r_id pairs
        if np.count_nonzero(
                negatives.rtable_id.values == rid) >= 1:  # if r_id takes part also in a negative predictions
            relatedTuples = negatives[
                negatives.rtable_id == rid]  # find all tuples where r_id participates in a negative prediction
            for curr_lid in relatedTuples.ltable_id.values:  # collect all other l_ids that also are part of the negative prediction
                # add a new triangle with l_id1, a r_id1 participating in a positive prediction (with l_id1), and another l_id2 that participates in a negative prediction with r_id1
                triangles.append((lid, rid, curr_lid))
        if np.count_nonzero(
                negatives.ltable_id.values == lid) >= 1:  # dual but starting from l_id1 in positive prediction with r_id1, looking for r_id2s where l_id participates in a negative prediction
            relatedTuples = negatives[negatives.ltable_id == lid]
            for curr_rid in relatedTuples.rtable_id.values:
                triangles.append((rid, lid, curr_rid))
    return triangles, sourcesmap


def __getRecords(sourcesMap, triangleIds, lprefix, rprefix):
    triangle = []
    for sourceid_recordid in triangleIds:
        split = str(sourceid_recordid).split("@")
        source_index = int(split[0])
        if source_index == 0:
            prefix = lprefix
        else:
            prefix = rprefix
        currentSource = sourcesMap[source_index]
        currentRecordId = int(split[1])
        currentRecord = currentSource[currentSource[prefix + 'id'] == currentRecordId].iloc[0]
        triangle.append(currentRecord)
    return triangle


def createPerturbationsFromTriangle(triangleIds, sourcesMap, attributes, maxLenAttributeSet, classToExplain, lprefix,
                                    rprefix):
    # generate power set of attributes
    allAttributesSubsets = list(_powerset(attributes, maxLenAttributeSet, maxLenAttributeSet))
    triangle = __getRecords(sourcesMap, triangleIds, lprefix, rprefix)  # get triangle values
    perturbations = []
    perturbedAttributes = []
    droppedValues = []
    copiedValues = []
    for subset in allAttributesSubsets:  # iterate over the attribute power set
        dv = []
        cv = []
        if classToExplain == 1:
            newRecord = triangle[0].copy()  # copy the l1 tuple
            if not all(elem in newRecord.index.to_list() for elem in subset):
                continue
            perturbedAttributes.append(subset)
            for att in subset:
                dv.append(newRecord[att])
                cv.append(triangle[2][att])
                newRecord[att] = triangle[2][
                    att]  # copy the value for the given attribute from l2 of no-match l2, r1 pair into l1
            perturbations.append(newRecord)  # append the new record
        else:
            newRecord = triangle[2].copy()  # copy the l2 tuple
            if not all(elem in newRecord.index.to_list() for elem in subset):
                continue
            perturbedAttributes.append(subset)
            for att in subset:
                dv.append(newRecord[att])
                cv.append(triangle[0][att])
                newRecord[att] = triangle[0][
                    att]  # copy the value for the given attribute from l1 of match l1, r1 pair into l2
            perturbations.append(newRecord)  # append the new record
        droppedValues.append(dv)
        copiedValues.append(cv)
    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r2 = triangle[1].copy()
    r2_copy = [r2] * len(perturbations_df)
    r2_df = pd.DataFrame(r2_copy, index=np.arange(len(perturbations)))
    if perturbations_df.columns[0].startswith(lprefix):
        allPerturbations = pd.concat([perturbations_df, r2_df], axis=1)
    else:
        allPerturbations = pd.concat([r2_df, perturbations_df], axis=1)
    allPerturbations = allPerturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    allPerturbations['alteredAttributes'] = perturbedAttributes
    allPerturbations['droppedValues'] = droppedValues
    allPerturbations['copiedValues'] = copiedValues

    return allPerturbations


def createTokenPerturbationsFromTriangle(triangleIds, sourcesMap, attributes, maxLenAttributeSet, classToExplain, lprefix,
                                    rprefix, idx, check=False):
    # generate power set of attributes
    allAttributesSubsets = list(_powerset(attributes, maxLenAttributeSet, maxLenAttributeSet))
    triangle = __getRecords(sourcesMap, triangleIds, lprefix, rprefix)  # get triangle values
    perturbations = []
    perturbedAttributes = []
    droppedValues = []
    copiedValues = []
    for subset in allAttributesSubsets:  # iterate over the attribute power set
        if classToExplain == 1:
            support = triangle[2].copy()
            free = triangle[0].copy()
        else:
            support = triangle[0].copy()
            free = triangle[2].copy()

        if not all(elem.split('__')[0] in free.index.to_list() for elem in subset):
            continue

        repls = []
        aa = []
        replacements = dict()
        for tbc in subset:  # iterate over the attribute:tokens
            affected_attribute = tbc.split('__')[0]  # attribute to be affected
            aa.append(affected_attribute)
            if affected_attribute in support.index:
                replacement_value = support[affected_attribute]
                replacement_tokens = replacement_value.split(' ')
                replacements[affected_attribute] = replacement_tokens
                for rt in replacement_tokens:
                    repls.append('__'.join([affected_attribute, rt]))

        all_rt_combs = list(_powerset(repls, maxLenAttributeSet, maxLenAttributeSet))
        filtered_combs = []
        for comb in all_rt_combs:
            naas = []
            for rt in comb:
                naas.append(rt.split('__')[0])
            if aa == naas:
                filtered_combs.append(comb)

        for comb in filtered_combs:
            naas = []
            for rt in comb:
                naas.append(rt.split('__')[0])
            newRecord = free.copy()
            dv = []
            cv = []
            affected_attributes = []
            ic = 0
            for tbc in subset:  # iterate over the attribute:tokens
                affected_attribute = tbc.split('__')[0]  # attribute to be affected
                affected_token = tbc.split('__')[1]  # token to be replaced
                if affected_attribute in support.index and affected_attribute in replacements \
                        and len(replacements[affected_attribute]) > 0:
                    replacement_token = comb[ic].split('__')[1]
                    newRecord[affected_attribute] = newRecord[affected_attribute].replace(affected_token,
                                                                                          replacement_token)
                    dv.append(affected_token)
                    cv.append(replacement_token)
                    affected_attributes.append(tbc)
                    ic +=1
            if not all(newRecord == free) and len(dv) == maxLenAttributeSet:
                good = True
                if check:
                    for i in range(len(cv)):
                        original_text = free[aa[i]]
                        split = original_text.split(dv[i])
                        prev_text = split[0].strip().split(' ')
                        prev_token = prev_text[len(prev_text) - 1]
                        prev_span = prev_token + ' ' + cv[i]
                        next_text = split[1].strip().split(' ')
                        next_token = next_text[0]
                        next_span = cv[i] + ' ' + next_token
                        good = good and (prev_token == '' or prev_span.strip() in idx) \
                               or (next_token == '' or next_span.strip() in idx)
                if good:
                    droppedValues.append(dv)
                    copiedValues.append(cv)
                    perturbations.append(newRecord)
                    perturbedAttributes.append(subset)


    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r2 = triangle[1].copy()
    r2_copy = [r2] * len(perturbations_df)
    r2_df = pd.DataFrame(r2_copy, index=np.arange(len(perturbations)))
    if perturbations_df.columns[0].startswith(lprefix):
        allPerturbations = pd.concat([perturbations_df, r2_df], axis=1)
    else:
        allPerturbations = pd.concat([r2_df, perturbations_df], axis=1)
    allPerturbations = allPerturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    allPerturbations['alteredAttributes'] = perturbedAttributes
    allPerturbations['droppedValues'] = droppedValues
    allPerturbations['copiedValues'] = copiedValues

    return allPerturbations


def check_properties(triangle, sourcesMap, predict_fn):
    try:
        t1 = triangle[0].split('@')
        t2 = triangle[1].split('@')
        t3 = triangle[2].split('@')
        if int(t1[0]) == 0:
            u = pd.DataFrame(sourcesMap.get(int(t1[0])).iloc[int(t1[1])]).transpose()
            v = pd.DataFrame(sourcesMap.get(int(t2[0])).iloc[int(t2[1])]).transpose()
            w1 = pd.DataFrame(sourcesMap.get(int(t3[0])).iloc[int(t3[1])]).transpose()
            u1 = u.copy()
            v1 = v.copy()
            w = w1.copy()

        else:
            u = pd.DataFrame(sourcesMap.get(int(t2[0])).iloc[int(t2[1])]).transpose()
            v = pd.DataFrame(sourcesMap.get(int(t1[0])).iloc[int(t1[1])]).transpose()
            w = pd.DataFrame(sourcesMap.get(int(t3[0])).iloc[int(t3[1])]).transpose()
            u1 = u.copy()
            v1 = v.copy()
            w1 = w.copy()

        _renameColumnsWithPrefix('ltable_', u)
        _renameColumnsWithPrefix('rtable_', u1)
        _renameColumnsWithPrefix('rtable_', v)
        _renameColumnsWithPrefix('ltable_', v1)
        _renameColumnsWithPrefix('rtable_', w)
        _renameColumnsWithPrefix('ltable_', w1)

        records = pd.concat([
            # identity
            pd.concat([u.reset_index(), u1.reset_index()], axis=1),
            pd.concat([v1.reset_index(), v.reset_index()], axis=1),
            pd.concat([w1.reset_index(), w.reset_index()], axis=1),

            # symmetry
            pd.concat([u.reset_index(), v.reset_index()], axis=1),
            pd.concat([v1.reset_index(), u1.reset_index()], axis=1),
            pd.concat([u.reset_index(), w.reset_index()], axis=1),
            pd.concat([w1.reset_index(), u1.reset_index()], axis=1),
            pd.concat([v1.reset_index(), w.reset_index()], axis=1),
            pd.concat([w1.reset_index(), v.reset_index()], axis=1),

            # transitivity
            pd.concat([u.reset_index(), v.reset_index()], axis=1),
            pd.concat([v1.reset_index(), w.reset_index()], axis=1),
            pd.concat([u.reset_index(), w.reset_index()], axis=1)
        ])

        predictions = np.argmax(predict_fn(records)[['nomatch_score', 'match_score']].values, axis=1)

        identity = predictions[0] == 1 and predictions[1] == 1 and predictions[2] == 1

        symmetry = predictions[3] == predictions[4] and predictions[5] == predictions[6] \
                   and predictions[7] == predictions[8]

        p1 = predictions[9]
        p2 = predictions[10]
        p3 = predictions[11]

        matches = 0
        non_matches = 0
        if p1 == 1:
            matches += 1
        else:
            non_matches += 1
        if p2 == 1:
            matches += 1
        else:
            non_matches += 1
        if p3 == 1:
            matches += 1
        else:
            non_matches += 1
        transitivity = matches == 3 or non_matches == 3 or (matches == 1 and non_matches == 2)

        return identity, symmetry, transitivity
    except:
        return False, False, False


def perturb_predict_token(allTriangles, attributes, class_to_explain, attr_length, predict_fn,
                          sourcesMap, lprefix, rprefix, token_combinations=3):
    all_predictions = pd.DataFrame()
    rankings = []
    flippedPredictions = []
    # lattice stratified predictions
    idx = None
    for att in sourcesMap[0]:
        if att not in ['ltable_id', 'id', 'rtable_id', 'label']:
            ar2 = np.concatenate(list(sourcesMap[0][att].apply(lambda row: [r for r in (' '.join(t) for t in nltk.bigrams(row.split(' ')))]).values))
            ar1 = np.concatenate(list(sourcesMap[0][att].apply(lambda row: [r for r in (' '.join(t) for t in nltk.ngrams(row.split(' '), 1))]).values))
            ar = np.concatenate([ar1, ar2])
            if idx is None:
                idx = ar
            else:
                idx = np.concatenate((ar, idx), axis=0)
    for att in sourcesMap[1]:
        if att not in ['ltable_id', 'id', 'rtable_id', 'label']:
            ar2 = np.concatenate(list(sourcesMap[1][att].apply(
                lambda row: [r for r in (' '.join(t) for t in nltk.bigrams(row.split(' ')))]).values))
            ar1 = np.concatenate(list(sourcesMap[1][att].apply(
                lambda row: [r for r in (' '.join(t) for t in nltk.ngrams(row.split(' '), 1))]).values))
            ar = np.concatenate([ar1, ar2])
            idx = np.concatenate((ar, idx), axis=0)

    all_good = False
    for a in range(1, token_combinations):
        t_i = 0
        perturbations = []
        for triangle in tqdm(allTriangles):
            try:
                currentPerturbations = createTokenPerturbationsFromTriangle(triangle, sourcesMap, attributes, a,
                                                                       class_to_explain, lprefix, rprefix, idx)
                currentPerturbations['triangle'] = ' '.join(triangle)
                perturbations.append(currentPerturbations)
            except Exception as e:
                pass
            t_i += 1

        try:
            perturbations_df = pd.concat(perturbations, ignore_index=True)
        except:
            perturbations_df = pd.DataFrame(perturbations)
        if len(perturbations_df) == 0 or 'alteredAttributes' not in perturbations_df.columns:
            continue
        currPerturbedAttr = perturbations_df.alteredAttributes.values
        if a != attr_length and not all_good:
            predictions = predict_fn(
                perturbations_df.drop(['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle'], axis=1))
            predictions = pd.concat(
                [predictions, perturbations_df[['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle']]],
                axis=1)
            all_predictions = pd.concat([all_predictions, predictions])
            proba = predictions[['nomatch_score', 'match_score']].values

            curr_flippedPredictions = predictions[proba[:, class_to_explain] < 0.5]
        else:
            proba = pd.DataFrame(columns=['nomatch_score', 'match_score'])

            if class_to_explain == 0:
                proba.loc[:, 'nomatch_score'] = np.zeros([len(perturbations_df)])
                proba.loc[:, 'match_score'] = np.ones([len(perturbations_df)])
            else:
                proba.loc[:, 'match_score'] = np.zeros([len(perturbations_df)])
                proba.loc[:, 'nomatch_score'] = np.ones([len(perturbations_df)])

            curr_flippedPredictions = pd.concat([perturbations_df.copy(), proba], axis=1)
            proba = proba.values
            all_predictions = pd.concat([all_predictions, curr_flippedPredictions])

        flippedPredictions.append(curr_flippedPredictions)

        ranking = getAttributeRanking(proba, currPerturbedAttr, class_to_explain)
        rankings.append(ranking)

        if len(curr_flippedPredictions) == len(perturbations_df):
            logging.info(f'skipped predictions at depth {a}')
            all_good = True
        else:
            logging.debug(f'predicted depth {a}')
    try:
        flippedPredictions_df = pd.concat(flippedPredictions, ignore_index=True)
    except:
        flippedPredictions_df = pd.DataFrame(flippedPredictions)
    return flippedPredictions_df, rankings, all_predictions


def explain_samples(dataset: pd.DataFrame, sources: list, predict_fn: callable, lprefix, rprefix,
                    class_to_explain: int, attr_length: int, check: bool = False,
                    discard_bad: bool = False, return_top: bool = False,
                    persist_predictions: bool = False, token: bool = False):
    _renameColumnsWithPrefix(lprefix, sources[0])
    _renameColumnsWithPrefix(rprefix, sources[1])

    if token:
        record = pd.DataFrame(dataset.iloc[0]).T
        # we need to map records from series of attributes into series of tokens, attribute names are mapped to "original" token names
        attributes = []
        for column in record.columns:
            if column not in ['label', 'id', lprefix+'id', rprefix+'id']:
                tokens = str(record[column].values[0]).split(' ')
                for t in tokens:
                    attributes.append(column+'__'+t)
        allTriangles, sourcesMap = getMixedTriangles(dataset, sources)
        if len(allTriangles) > 0:
            flipped_predictions, rankings, all_predictions = perturb_predict_token(allTriangles, attributes,
                                                                             class_to_explain,
                                                                             attr_length, predict_fn, sourcesMap,
                                                                             lprefix,
                                                                             rprefix, token_combinations=int(len(attributes)/2))
            if persist_predictions:
                all_predictions.to_csv('predictions.csv')
            explanation = aggregateRankings(rankings, lenTriangles=len(allTriangles), attr_length=attr_length)

            flips = len(flipped_predictions) + len(allTriangles)
            saliency = dict()

            for ranking in rankings:
                for k, v in ranking.items():
                    for a in k:
                        if a not in saliency:
                            saliency[a] = len(allTriangles) / flips
                        saliency[a] += v / flips

            if len(explanation) > 0:
                if len(flipped_predictions) > 0:
                    flipped_predictions['attr_count'] = flipped_predictions.alteredAttributes.astype(str) \
                        .str.split(',').str.len()
                    flipped_predictions = flipped_predictions.sort_values(by=['attr_count'])
                if return_top:
                    series = cf_summary(explanation)
                    filtered_exp = series
                else:
                    filtered_exp = explanation

                return saliency, filtered_exp, flipped_predictions, allTriangles
            else:
                logging.warning(f'empty explanation !?')
                return dict(), pd.DataFrame(), pd.DataFrame(), []
        else:
            logging.warning(f'empty triangles !?')
            return dict(), pd.DataFrame(), pd.DataFrame(), []


    else:
        attributes = [col for col in list(sources[0]) if col not in [lprefix + 'id']]
        attributes += [col for col in list(sources[1]) if col not in [rprefix + 'id']]

        allTriangles, sourcesMap = getMixedTriangles(dataset, sources)
        if len(allTriangles) > 0:
            flipped_predictions, rankings, all_predictions = perturb_predict(allTriangles, attributes, check,
                                                                               class_to_explain, discard_bad,
                                                                               attr_length, predict_fn, sourcesMap, lprefix,
                                                                               rprefix)
            if persist_predictions:
                all_predictions.to_csv('predictions.csv')
            explanation = aggregateRankings(rankings, lenTriangles=len(allTriangles), attr_length=attr_length)

            flips = len(flipped_predictions) + len(allTriangles)
            saliency = dict()
            for a in dataset.columns:
                if (a.startswith(lprefix) or a.startswith(rprefix)) and not (a == lprefix + 'id' or a == rprefix + 'id'):
                    saliency[a] = len(allTriangles) / flips # all attributes have a flip for the entire attribute set A

            for ranking in rankings:
                for k, v in ranking.items():
                    for a in k:
                        saliency[a] += v / flips

            if len(explanation) > 0:
                if len(flipped_predictions) > 0:
                    flipped_predictions['attr_count'] = flipped_predictions.alteredAttributes.astype(str) \
                        .str.split(',').str.len()
                    flipped_predictions = flipped_predictions.sort_values(by=['attr_count'])
                if return_top:
                    series = cf_summary(explanation)
                    filtered_exp = series
                else:
                    filtered_exp = explanation

                return saliency, filtered_exp, flipped_predictions, allTriangles
            else:
                return dict(), [], pd.DataFrame(), []
        else:
            logging.warning(f'empty triangles !?')
            return dict(), [], pd.DataFrame(), []


def cf_summary(explanation):
    sorted_attr_pairs = explanation.sort_values(ascending=False)
    explanations = sorted_attr_pairs.loc[sorted_attr_pairs.values == sorted_attr_pairs.values.max()]
    filtered = [i for i in explanations.keys() if
                not any(all(c in i for c in b) and len(b) < len(i) for b in explanations.keys())]
    filtered_exp = {}
    for te in filtered:
        filtered_exp[te] = explanations[te]
    series = pd.Series(index=filtered_exp.keys(), data=filtered_exp.values())
    return series


def perturb_predict(allTriangles, attributes, check, class_to_explain, discard_bad, attr_length, predict_fn,
                    sourcesMap, lprefix, rprefix, monotonicity=True):
    if monotonicity:
        all_predictions = pd.DataFrame()
        rankings = []
        transitivity = True
        flippedPredictions = []
        # lattice stratified predictions
        all_good = False
        for a in range(1, attr_length):
            t_i = 0
            perturbations = []
            for triangle in tqdm(allTriangles):
                try:
                    if check:
                        identity, symmetry, transitivity = check_properties(triangle, sourcesMap, predict_fn)
                        allTriangles[t_i] = allTriangles[t_i] + (identity, symmetry, transitivity,)
                    if check and discard_bad and not transitivity:
                        continue
                    currentPerturbations = createPerturbationsFromTriangle(triangle, sourcesMap, attributes, a,
                                                                           class_to_explain, lprefix, rprefix)
                    currentPerturbations['triangle'] = ' '.join(triangle)
                    perturbations.append(currentPerturbations)
                except:
                    allTriangles[t_i] = allTriangles[t_i] + (False, False, False,)
                    pass
                t_i += 1

            try:
                perturbations_df = pd.concat(perturbations, ignore_index=True)
            except:
                perturbations_df = pd.DataFrame(perturbations)
            if len(perturbations_df) == 0 or 'alteredAttributes' not in perturbations_df.columns:
                continue
            currPerturbedAttr = perturbations_df.alteredAttributes.values
            if a != attr_length and not all_good:
                predictions = predict_fn(perturbations_df.drop(['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle'], axis=1))
                predictions = pd.concat([predictions, perturbations_df[['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle']]], axis=1)
                all_predictions = pd.concat([all_predictions, predictions])
                proba = predictions[['nomatch_score', 'match_score']].values

                curr_flippedPredictions = predictions[proba[:, class_to_explain] < 0.5]
            else:
                proba = pd.DataFrame(columns=['nomatch_score', 'match_score'])

                if class_to_explain == 0:
                    proba.loc[:, 'nomatch_score'] = np.zeros([len(perturbations_df)])
                    proba.loc[:, 'match_score'] = np.ones([len(perturbations_df)])
                else:
                    proba.loc[:, 'match_score'] = np.zeros([len(perturbations_df)])
                    proba.loc[:, 'nomatch_score'] = np.ones([len(perturbations_df)])

                curr_flippedPredictions = pd.concat([perturbations_df.copy(), proba], axis=1)
                proba = proba.values

            flippedPredictions.append(curr_flippedPredictions)
            ranking = getAttributeRanking(proba, currPerturbedAttr, class_to_explain)
            rankings.append(ranking)

            if len(curr_flippedPredictions) == len(perturbations_df):
                logging.info(f'skipped predictions at depth {a}')
                all_good = True
            else:
                logging.debug(f'predicted depth {a}')
        try:
            flippedPredictions_df = pd.concat(flippedPredictions, ignore_index=True)
        except:
            flippedPredictions_df = pd.DataFrame(flippedPredictions)
        return flippedPredictions_df, rankings, all_predictions
    else:
        rankings = []
        transitivity = True
        flippedPredictions = []
        t_i = 0
        perturbations = []
        for triangle in tqdm(allTriangles):
            try:
                if check:
                    identity, symmetry, transitivity = check_properties(triangle, sourcesMap, predict_fn)
                    allTriangles[t_i] = allTriangles[t_i] + (identity, symmetry, transitivity,)
                if check and discard_bad and not transitivity:
                    continue
                currentPerturbations = createPerturbationsFromTriangle(triangle, sourcesMap, attributes,
                                                                       attr_length,
                                                                       class_to_explain, lprefix, rprefix)
                perturbations.append(currentPerturbations)
            except:
                allTriangles[t_i] = allTriangles[t_i] + (False, False, False,)
                pass
            t_i += 1
        try:
            perturbations_df = pd.concat(perturbations, ignore_index=True)
        except:
            perturbations_df = pd.DataFrame(perturbations)
        currPerturbedAttr = perturbations_df.alteredAttributes.values
        predictions = predict_fn(perturbations_df)
        predictions = predictions.drop(columns=['alteredAttributes'])
        proba = predictions[['nomatch_score', 'match_score']].values
        curr_flippedPredictions = perturbations_df[proba[:, class_to_explain] < 0.5]
        flippedPredictions.append(curr_flippedPredictions)
        ranking = getAttributeRanking(proba, currPerturbedAttr, class_to_explain)
        rankings.append(ranking)
        try:
            flippedPredictions_df = pd.concat(flippedPredictions, ignore_index=True)
        except:
            flippedPredictions_df = pd.DataFrame(flippedPredictions)
        return flippedPredictions_df, rankings, predictions


# for each prediction, if the original class is flipped, set the rank of the altered attributes to 1
def getAttributeRanking(proba: np.ndarray, alteredAttributes: list, originalClass: int):
    attributeRanking = {k: 0 for k in alteredAttributes}
    for i, prob in enumerate(proba):
        if float(prob[originalClass]) < 0.5:
            attributeRanking[alteredAttributes[i]] += 1
    return attributeRanking


# MaxLenAttributeSet is the max len of perturbed attributes we want to consider
# for each ranking, sum  the rank of each altered attribute
# then normalize the aggregated rank wrt the no. of triangles
def aggregateRankings(ranking_l: list, lenTriangles: int, attr_length: int):
    aggregateRanking = defaultdict(int)
    for ranking in ranking_l:
        for altered_attr in ranking.keys():
            if len(altered_attr) <= attr_length:
                aggregateRanking[altered_attr] += ranking[altered_attr]
    aggregateRanking_normalized = {k: (v / lenTriangles) for (k, v) in aggregateRanking.items()}

    alteredAttr = list(map(lambda t: "/".join(t), aggregateRanking_normalized.keys()))
    return pd.Series(data=list(aggregateRanking_normalized.values()), index=alteredAttr)
