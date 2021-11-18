import logging
from collections import defaultdict
from functools import partialmethod
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from certa.utils import diff

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def __getCorrectPredictions(dataset, model, predict_fn):
    predictions = predict_fn(dataset, model, ['label', 'id'])
    tp_group = dataset[(predictions[:, 1] >= 0.5) & (dataset['label'] == 1)]
    tn_group = dataset[(predictions[:, 0] >= 0.5) & (dataset['label'] == 0)]
    correctPredictions = pd.concat([tp_group, tn_group])
    return correctPredictions


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
    # if not 'ltable_id' in dataset_c.columns:
    dataset_c['ltable_id'] = list(map(lambda lrid: str(lrid).split("#")[0], dataset_c.id.values))
    # if not 'rtable_id' in dataset_c.columns:
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


# Not used for now
def getNegativeTriangles(dataset, sources):
    triangles = []
    dataset_c = dataset.copy()
    dataset_c['ltable_id'] = list(map(lambda lrid: int(str(lrid).split("#")[0]), dataset_c.id.values))
    dataset_c['rtable_id'] = list(map(lambda lrid: int(str(lrid).split("#")[1]), dataset_c.id.values))
    negatives = dataset_c[dataset_c.label == 0]
    l_neg_ids = negatives.ltable_id.values
    r_neg_ids = negatives.rtable_id.values
    for lid, rid in zip(l_neg_ids, r_neg_ids):
        if np.count_nonzero(r_neg_ids == rid) >= 2:
            relatedTuples = negatives[negatives.rtable_id == rid]
            for curr_lid in relatedTuples.ltable_id.values:
                if curr_lid != lid:
                    triangles.append((sources[0].iloc[lid], sources[1].iloc[rid], sources[0].iloc[curr_lid]))
        if np.count_nonzero(l_neg_ids == lid) >= 2:
            relatedTuples = negatives[negatives.ltable_id == lid]
            for curr_rid in relatedTuples.rtable_id.values:
                if curr_rid != rid:
                    triangles.append((sources[1].iloc[rid], sources[0].iloc[lid], sources[1].iloc[curr_rid]))
    return triangles


# not used for now
def getPositiveTriangles(dataset, sources):
    triangles = []
    dataset_c = dataset.copy()
    dataset_c['ltable_id'] = list(map(lambda lrid: int(str(lrid).split("#")[0]), dataset_c.id.values))
    dataset_c['rtable_id'] = list(map(lambda lrid: int(str(lrid).split("#")[1]), dataset_c.id.values))
    positives = dataset_c[dataset_c.label == 1]
    l_pos_ids = positives.ltable_id.values
    r_pos_ids = positives.rtable_id.values
    for lid, rid in zip(l_pos_ids, r_pos_ids):
        if np.count_nonzero(l_pos_ids == rid) >= 2:
            relatedTuples = positives[positives.rtable_id == rid]
            for curr_lid in relatedTuples.ltable_id.values:
                if curr_lid != lid:
                    triangles.append((sources[0].iloc[lid], sources[1].iloc[rid], sources[0].iloc[curr_lid]))
        if np.count_nonzero(l_pos_ids == lid) >= 2:
            relatedTuples = positives[positives.ltable_id == lid]
            for curr_rid in relatedTuples.rtable_id.values:
                if curr_rid != rid:
                    triangles.append((sources[1].iloc[rid], sources[0].iloc[lid], sources[1].iloc[curr_rid]))
    return triangles


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
    # _renameColumnsWithPrefix(rprefix, r2_df)
    # _renameColumnsWithPrefix(lprefix, perturbations_df)
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


def check_transitivity_text(model, predict_fn, u, v, v1, w, strict: bool = False):
    p1 = \
        predict_fn(pd.concat([u.reset_index(), v.reset_index()], axis=1), model)[
            ['nomatch_score', 'match_score']].values[0]
    p2 = \
        predict_fn(pd.concat([v1.reset_index(), w.reset_index()], axis=1), model)[
            ['nomatch_score', 'match_score']].values[
            0]
    p3 = \
        predict_fn(pd.concat([u.reset_index(), w.reset_index()], axis=1), model)[
            ['nomatch_score', 'match_score']].values[0]
    if strict:
        return p1[1] >= 0.5 and p2[0] >= 0.5 and p3[0] >= 0.5
    else:
        matches = 0
        non_matches = 0
        if p1[1] >= 0.5:
            matches += 1
        else:
            non_matches += 1
        if p2[1] >= 0.5:
            matches += 1
        else:
            non_matches += 1
        if p3[1] >= 0.5:
            matches += 1
        else:
            non_matches += 1
        return matches == 3 or non_matches == 3 or (matches == 1 and non_matches == 2)


def explainSamples(dataset: pd.DataFrame, sources: list, predict_fn: callable, lprefix, rprefix,
                   class_to_explain: int, attr_length: int, check: bool = False,
                   discard_bad: bool = False, return_top: bool = False,
                   persist_predictions: bool = False):
    _renameColumnsWithPrefix(lprefix, sources[0])
    _renameColumnsWithPrefix(rprefix, sources[1])

    attributes = [col for col in list(sources[0]) if col not in [lprefix + 'id']]
    attributes += [col for col in list(sources[1]) if col not in [rprefix + 'id']]

    allTriangles, sourcesMap = getMixedTriangles(dataset, sources)
    if len(allTriangles) > 0:
        flippedPredictions_df, rankings, all_predictions = perturb_predict(allTriangles, attributes, check,
                                                                           class_to_explain, discard_bad,
                                                                           attr_length, predict_fn, sourcesMap, lprefix,
                                                                           rprefix)

        if persist_predictions:
            all_predictions.to_csv('lattices.csv', mode='a')
        explanation = aggregateRankings(rankings, lenTriangles=len(allTriangles),
                                        attr_length=attr_length)

        if len(explanation) > 0:
            if return_top:
                series = cf_summary(explanation)
                filtered_exp = series
            else:
                filtered_exp = explanation

            return filtered_exp, flippedPredictions_df, allTriangles
        else:
            return [], pd.DataFrame(), []
    else:
        logging.warning(f'empty triangles !?')
        return [], pd.DataFrame(), []


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
                predictions = predict_fn(perturbations_df)
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
        return flippedPredictions_df, rankings, []
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
        return flippedPredictions_df, rankings


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


def createTokenPerturbationsFromTriangle(triangleIds, sourcesMap, attributes, attr_length, classToExplain,
                                         lprefix='ltable_', rprefix='rtable_', use_tokens: bool = False):
    # generate power set of attributes
    allAttributesSubsets = list(_powerset(attributes, 1, 1))
    triangle = __getRecords(sourcesMap, triangleIds[:3], lprefix, rprefix)  # get triangle values
    perturbations = []
    perturbedAttributes = []
    diffs = []
    for subset in allAttributesSubsets:  # iterate over the attribute power set
        if classToExplain == 1:
            for i in range(1, 10):
                newRecord1 = triangle[0].copy()  # copy the l1 tuple
                newRecord2 = triangle[0].copy()  # copy the l1 tuple
                for att in subset:
                    val = str(triangle[2][
                                  att])  # copy the value for the given attribute from l2 of no-match l2, r1 pair into l1
                    values = val.split()
                    val_cut = int(len(values) * i / 10)
                    old_val1 = newRecord1[att].split()
                    rec_cut = int(len(old_val1) * i / 10)

                    # generate new values with prefix / suffix dropped
                    new_val1 = " ".join(values[val_cut:])
                    newRecord1[att] = " ".join(old_val1[:rec_cut]) + new_val1
                    perturbations.append(newRecord1[:])  # append the new record
                    diffs.append(diff(" ".join(old_val1), newRecord1[att]))
                    perturbedAttributes.append(subset)

                    new_val2 = " ".join(values[:val_cut])
                    old_val2 = newRecord2[att].split()
                    newRecord2[att] = new_val2 + " ".join(old_val2[rec_cut:])
                    perturbations.append(newRecord2[:])  # append the new record
                    diffs.append(diff(" ".join(old_val2), newRecord2[att]))
                    perturbedAttributes.append(subset)
                # perturbations.append(newRecord1)  # append the new record
                # perturbations.append(newRecord2)  # append the new record
        else:
            for i in range(1, 10):
                newRecord1 = triangle[2].copy()  # copy the l2 tuple
                newRecord2 = triangle[2].copy()  # copy the l2 tuple
                for att in subset:
                    val = str(triangle[0][
                                  att])  # copy the value for the given attribute from l1 of no-match l1, r1 pair into l2
                    values = val.split()
                    val_cut = int(len(values) * i / 10)
                    old_val1 = str(newRecord1[att]).split()
                    rec_cut = int(len(old_val1) * i / 10)

                    # generate new values with prefix / suffix dropped
                    new_val1 = " ".join(values[val_cut:])
                    newRecord1[att] = " ".join(old_val1[:rec_cut]) + new_val1
                    perturbations.append(newRecord1[:])  # append the new record
                    diffs.append(diff(" ".join(old_val1), newRecord1[att]))
                    perturbedAttributes.append(subset)

                    new_val2 = " ".join(values[:val_cut])
                    old_val2 = str(newRecord2[att]).split()
                    newRecord2[att] = new_val2 + " ".join(old_val2[rec_cut:])
                    perturbations.append(newRecord2[:])  # append the new record
                    diffs.append(diff(" ".join(old_val2), newRecord2[att]))
                    perturbedAttributes.append(subset)
                # perturbations.append(newRecord1)  # append the new record
                # perturbations.append(newRecord2)  # append the new record
    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r2 = triangle[1].copy()
    r2_copy = [r2] * len(perturbations_df)
    r2_df = pd.DataFrame(r2_copy, index=np.arange(len(perturbations)))
    _renameColumnsWithPrefix(rprefix, r2_df)
    _renameColumnsWithPrefix(lprefix, perturbations_df)
    allPerturbations = pd.concat([perturbations_df, r2_df], axis=1)
    allPerturbations = allPerturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    allPerturbations['alteredAttributes'] = perturbedAttributes
    allPerturbations['alteredTokens'] = diffs
    return allPerturbations
