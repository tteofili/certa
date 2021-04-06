import operator

import pandas as pd
from itertools import chain, combinations
from collections import defaultdict
import numpy as np
import random as rd
import math
import string
import os
from tqdm import tqdm


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
    # a triangle is a triple <u, v, w> where <u, v> is a match and <v, w> is a non-match
    triangles = []
    # to not alter original dataset
    dataset_c = dataset.copy()
    sourcesmap = {}
    # the id is so composed: lsourcenumber@id#rsourcenumber@id
    for i in range(len(sources)):
        sourcesmap[i] = sources[i]
    if not 'ltable_id' in dataset_c.columns:
        dataset_c['ltable_id'] = list(map(lambda lrid: str(lrid).split("#")[0], dataset_c.id.values))
    if not 'rtable_id' in dataset_c.columns:
        dataset_c['rtable_id'] = list(map(lambda lrid: str(lrid).split("#")[1], dataset_c.id.values))
    positives = dataset_c[dataset_c.label == 1]  # match classified samples
    negatives = dataset_c[dataset_c.label == 0]  # no-match classified samples
    l_pos_ids = positives.ltable_id.astype('str').values  # left ids of positive samples
    r_pos_ids = positives.rtable_id.astype('str').values  # right ids of positive samples
    for lid, rid in zip(l_pos_ids, r_pos_ids):  # iterate through l_id, r_id pairs
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


def __getRecords(sourcesMap, triangleIds):
    triangle = []
    for sourceid_recordid in triangleIds:
        currentSource = sourcesMap[int(str(sourceid_recordid).split("@")[0])]
        currentRecordId = int(str(sourceid_recordid).split("@")[1])
        currentRecord = currentSource[currentSource.id == currentRecordId].iloc[0]
        triangle.append(currentRecord)
    return triangle


def createPerturbationsFromTriangle(triangleIds, sourcesMap, attributes, maxLenAttributeSet, classToExplain,
                                    lprefix='ltable_', rprefix='rtable_'):
    # generate power set of attributes
    allAttributesSubsets = list(_powerset(attributes, 1, maxLenAttributeSet))
    triangle = __getRecords(sourcesMap, triangleIds)  # get triangle values
    perturbations = []
    perturbedAttributes = []
    for subset in allAttributesSubsets:  # iterate over the attribute power set
        perturbedAttributes.append(subset)
        if classToExplain == 1:
            newRecord = triangle[1].copy()  # copy the r1 tuple
            rightRecordId = triangleIds[1]
            for att in subset:
                newRecord[att] = triangle[2][att]  # copy the value for the given attribute from no-match l2 tuple into r1
            perturbations.append(newRecord)  # append the new record
        else:
            newRecord = triangle[2].copy()  # copy the l2 tuple
            rightRecordId = triangleIds[2]
            for att in subset:
                newRecord[att] = triangle[1][att]  # copy the value for the given attribute from match r1 tuple into l2
            perturbations.append(newRecord)  # append the new record
    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r1 = triangle[0].copy()
    r1_copy = [r1] * len(perturbations_df)
    r1_df = pd.DataFrame(r1_copy, index=np.arange(len(perturbations)))
    _renameColumnsWithPrefix(lprefix, r1_df)
    _renameColumnsWithPrefix(rprefix, perturbations_df)
    allPerturbations = pd.concat([r1_df, perturbations_df], axis=1)
    allPerturbations = allPerturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    allPerturbations['alteredAttributes'] = perturbedAttributes
    allPerturbations['originalRightId'] = rightRecordId
    return allPerturbations


def check_transitivity(triangle, sourcesMap, predict_fn, model):
    u = pd.DataFrame(sourcesMap.get(int(triangle[0].split('@')[0])).iloc[int(triangle[0].split('@')[1])]).transpose()
    v = pd.DataFrame(sourcesMap.get(int(triangle[1].split('@')[0])).iloc[int(triangle[1].split('@')[1])]).transpose()
    v1 = v.copy()
    w = pd.DataFrame(sourcesMap.get(int(triangle[2].split('@')[0])).iloc[int(triangle[2].split('@')[1])]).transpose()
    _renameColumnsWithPrefix('ltable_', u)
    _renameColumnsWithPrefix('rtable_', v)
    _renameColumnsWithPrefix('ltable_', v1)
    _renameColumnsWithPrefix('rtable_', w)
    return check_transitivity_text(model, predict_fn, u, v, v1, w)


def check_transitivity_text(model, predict_fn, u, v, v1, w):
    p1 = predict_fn(pd.concat([u.reset_index(), v.reset_index()], axis=1), model)[['nomatch_score', 'match_score']].values[0]
    p2 = predict_fn(pd.concat([v1.reset_index(), w.reset_index()], axis=1), model)[['nomatch_score', 'match_score']].values[0]
    p3 = predict_fn(pd.concat([u.reset_index(), w.reset_index()], axis=1), model)[['nomatch_score', 'match_score']].values[0]
    return p1[1] >= 0.5 and p2[0] >= 0.5 and p3[0] >= 0.5


def explainSamples(dataset: pd.DataFrame, sources: list, model, predict_fn: callable,
                   class_to_explain: int, maxLenAttributeSet: int, check: bool):
    attributes = [col for col in list(sources[0]) if col not in ['id']]
    allTriangles, sourcesMap = getMixedTriangles(dataset, sources)
    rankings = []
    flippedPredictions = []
    t_i = 0
    for triangle in tqdm(allTriangles):
        if check:
            pre_good = check_transitivity(triangle, sourcesMap, predict_fn, model)
            allTriangles[t_i] = allTriangles[t_i] + (pre_good, )
        currentPerturbations = createPerturbationsFromTriangle(triangle, sourcesMap, attributes, maxLenAttributeSet,
                                                               class_to_explain)
        currPerturbedAttr = currentPerturbations.alteredAttributes.values
        predictions = predict_fn(currentPerturbations, model)
        predictions = predictions.drop(columns=['alteredAttributes', 'originalRightId'])
        proba = predictions[['nomatch_score', 'match_score']].values
        curr_flippedPredictions = currentPerturbations[proba[:, class_to_explain] < 0.5]
        flippedPredictions.append(curr_flippedPredictions)
        ranking = getAttributeRanking(proba, currPerturbedAttr, class_to_explain)
        rankings.append(ranking)
        t_i += 1
    try:
        flippedPredictions_df = pd.concat(flippedPredictions, ignore_index=True)
    except:
        flippedPredictions_df = pd.DataFrame(flippedPredictions)
    explanation = aggregateRankings(rankings, lenTriangles=len(allTriangles),
                                           maxLenAttributeSet=maxLenAttributeSet)

    if len(explanation) > 0:
        sorted_attr_pairs = explanation.sort_values(ascending=False)
        explanations = sorted_attr_pairs.loc[sorted_attr_pairs.values == sorted_attr_pairs.values.max()]
        filtered = [i for i in explanations.keys() if
                    not any(all(c in i for c in b) and len(b) < len(i) for b in explanations.keys())]
        filtered_exp = defaultdict(int)
        for te in filtered:
            filtered_exp[te] = explanations[te]

        return filtered_exp, flippedPredictions_df, allTriangles
    else:
        return [], pd.DataFrame(), []


# for each prediction, if the original class is flipped, set the rank of the altered attributes to 1
def getAttributeRanking(proba: np.ndarray, alteredAttributes: list, originalClass: int):
    attributeRanking = {k: 0 for k in alteredAttributes}
    for i, prob in enumerate(proba):
        if prob[originalClass] < 0.5:
            attributeRanking[alteredAttributes[i]] += 1
    return attributeRanking


# MaxLenAttributeSet is the max len of perturbed attributes we want to consider
# for each ranking, sum  the rank of each altered attribute
# then normalize the aggregated rank wrt the no. of triangles
def aggregateRankings(ranking_l: list, lenTriangles: int, maxLenAttributeSet: int):
    aggregateRanking = defaultdict(int)
    for ranking in ranking_l:
        for altered_attr in ranking.keys():
            if len(altered_attr) <= maxLenAttributeSet:
                aggregateRanking[altered_attr] += ranking[altered_attr]
    aggregateRanking_normalized = {k: (v / lenTriangles) for (k, v) in aggregateRanking.items()}

    alteredAttr = list(map(lambda t: "/".join(t), aggregateRanking_normalized.keys()))
    return pd.Series(data=list(aggregateRanking_normalized.values()), index=alteredAttr)
