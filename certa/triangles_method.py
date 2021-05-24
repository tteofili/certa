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

from certa.utils import diff


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
    #if not 'ltable_id' in dataset_c.columns:
    dataset_c['ltable_id'] = list(map(lambda lrid: str(lrid).split("#")[0], dataset_c.id.values))
    #if not 'rtable_id' in dataset_c.columns:
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


def __getRecords(sourcesMap, triangleIds):
    triangle = []
    for sourceid_recordid in triangleIds:
        currentSource = sourcesMap[int(str(sourceid_recordid).split("@")[0])]
        currentRecordId = int(str(sourceid_recordid).split("@")[1])
        currentRecord = currentSource[currentSource.id == currentRecordId].iloc[0]
        triangle.append(currentRecord)
    return triangle


def createPerturbationsFromTriangle(triangleIds, sourcesMap, attributes, maxLenAttributeSet, classToExplain,
                                    lprefix='ltable_', rprefix='rtable_', use_tokens: bool = False):
    # generate power set of attributes
    allAttributesSubsets = list(_powerset(attributes, 1, maxLenAttributeSet))
    triangle = __getRecords(sourcesMap, triangleIds)  # get triangle values
    perturbations = []
    perturbedAttributes = []
    for subset in allAttributesSubsets:  # iterate over the attribute power set
        perturbedAttributes.append(subset)
        if classToExplain == 1:
            newRecord = triangle[0].copy()  # copy the l1 tuple
            for att in subset:
                newRecord[att] = triangle[2][att]  # copy the value for the given attribute from l2 of no-match l2, r1 pair into l1
            perturbations.append(newRecord)  # append the new record
        else:
            newRecord = triangle[2].copy()  # copy the l2 tuple
            for att in subset:
                newRecord[att] = triangle[0][att]  # copy the value for the given attribute from l1 of match l1, r1 pair into l2
            perturbations.append(newRecord)  # append the new record
    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r2 = triangle[1].copy()
    r2_copy = [r2] * len(perturbations_df)
    r2_df = pd.DataFrame(r2_copy, index=np.arange(len(perturbations)))
    _renameColumnsWithPrefix(rprefix, r2_df)
    _renameColumnsWithPrefix(lprefix, perturbations_df)
    allPerturbations = pd.concat([perturbations_df, r2_df], axis=1)
    allPerturbations = allPerturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    allPerturbations['alteredAttributes'] = perturbedAttributes

    return allPerturbations


def check_properties(triangle, sourcesMap, predict_fn, model):
    try:
        t1 = triangle[0].split('@')
        t2 = triangle[1].split('@')
        t3 = triangle[2].split('@')
        u = pd.DataFrame(sourcesMap.get(int(t1[0])).iloc[int(t1[1])]).transpose()
        v = pd.DataFrame(sourcesMap.get(int(t2[0])).iloc[int(t2[1])]).transpose()
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
        #identity
        pd.concat([u.reset_index(), u1.reset_index()], axis=1),
        pd.concat([v1.reset_index(), v.reset_index()], axis=1),
        pd.concat([w1.reset_index(), w.reset_index()], axis=1),

        #symmetry
        pd.concat([u.reset_index(), v.reset_index()], axis=1),
        pd.concat([v1.reset_index(), u1.reset_index()], axis=1),
        pd.concat([u.reset_index(), w.reset_index()], axis=1),
        pd.concat([w1.reset_index(), u1.reset_index()], axis=1),
        pd.concat([v1.reset_index(), w.reset_index()], axis=1),
        pd.concat([w1.reset_index(), v.reset_index()], axis=1),

        #transitivity
        pd.concat([u.reset_index(), v.reset_index()], axis=1),
        pd.concat([v1.reset_index(), w.reset_index()], axis=1),
        pd.concat([u.reset_index(), w.reset_index()], axis=1)
        ])

        predictions = np.argmax(predict_fn(records, model)[['nomatch_score', 'match_score']].values, axis=1)

        identity = predictions[0] == 1 and predictions[1] == 1 and predictions[2] == 1

        symmetry = predictions[3] == predictions[4] and predictions[5] == predictions[6] \
                   and predictions[7] == predictions[8]

        p1 = predictions[9]
        p2 = predictions[10]
        p3 = predictions[11]

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
        transitivity = matches == 3 or non_matches == 3 or (matches == 1 and non_matches == 2)

        return identity, symmetry, transitivity
    except:
        return False, False, False


def check_transitivity_text(model, predict_fn, u, v, v1, w, strict: bool = False):
    p1 = predict_fn(pd.concat([u.reset_index(), v.reset_index()], axis=1), model)[['nomatch_score', 'match_score']].values[0]
    p2 = predict_fn(pd.concat([v1.reset_index(), w.reset_index()], axis=1), model)[['nomatch_score', 'match_score']].values[0]
    p3 = predict_fn(pd.concat([u.reset_index(), w.reset_index()], axis=1), model)[['nomatch_score', 'match_score']].values[0]
    if strict:
        return p1[1] >= 0.5 and p2[0] >= 0.5 and p3[0] >= 0.5
    else :
        matches = 0
        non_matches = 0
        if p1[1] >= 0.5:
            matches += 1
        else:
            non_matches +=1
        if p2[1] >= 0.5:
            matches += 1
        else:
            non_matches +=1
        if p3[1] >= 0.5:
            matches += 1
        else:
            non_matches +=1
        return matches == 3 or non_matches == 3 or (matches == 1 and non_matches == 2)


def explainSamples(dataset: pd.DataFrame, sources: list, model, predict_fn: callable,
                   class_to_explain: int, maxLenAttributeSet: int, check: bool, tokens: bool = False,
                   discard_bad: bool = False):
    attributes = [col for col in list(sources[0]) if col not in ['id']]
    allTriangles, sourcesMap = getMixedTriangles(dataset, sources)
    rankings = []
    flippedPredictions = []
    t_i = 0
    transitivity = True
    for triangle in tqdm(allTriangles):
        try:
            if check:
                identity, symmetry, transitivity = check_properties(triangle, sourcesMap, predict_fn, model)
                allTriangles[t_i] = allTriangles[t_i] + (identity, symmetry, transitivity, )
            if check and discard_bad and not transitivity:
                continue
            currentPerturbations = createPerturbationsFromTriangle(triangle, sourcesMap, attributes, maxLenAttributeSet,
                                                                   class_to_explain)
            currPerturbedAttr = currentPerturbations.alteredAttributes.values
            predictions = predict_fn(currentPerturbations, model)
            predictions = predictions.drop(columns=['alteredAttributes'])
            proba = predictions[['nomatch_score', 'match_score']].values
            curr_flippedPredictions = currentPerturbations[proba[:, class_to_explain] < 0.5]
            flippedPredictions.append(curr_flippedPredictions)
            ranking = getAttributeRanking(proba, currPerturbedAttr, class_to_explain)
            rankings.append(ranking)
        except:
            allTriangles[t_i] = allTriangles[t_i] + (False, False, False, )
            pass
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

        if tokens:
            # todo: to be finished (weight by score, properly aggregate token rankings, better explanation (not string)
            token_filtered_exp = defaultdict(int)
            token_rankings = []
            token_flippedPredictions = []
            token_flippedPredictions_df = pd.DataFrame()
            for exp in filtered_exp:
                e_attrs = exp.split('/')
                e_score = explanation[exp]

                for triangle in tqdm(allTriangles):
                    currentTokenPerturbations = createTokenPerturbationsFromTriangle(triangle, sourcesMap, e_attrs, maxLenAttributeSet,
                                                                           class_to_explain)
                    currPerturbedAttr = currentTokenPerturbations[['alteredAttributes', 'alteredTokens']].apply(
                            lambda x: ':'.join(x.dropna().astype(str)), axis=1).values
                    predictions = predict_fn(currentTokenPerturbations, model)
                    predictions = predictions.drop(columns=['alteredAttributes', 'alteredTokens'])
                    proba = predictions[['nomatch_score', 'match_score']].values
                    curr_flippedPredictions = currentTokenPerturbations[proba[:, class_to_explain] < 0.5]
                    token_flippedPredictions.append(curr_flippedPredictions)
                    token_ranking = getAttributeRanking(proba, currPerturbedAttr, class_to_explain)
                    token_rankings.append(token_ranking)

                try:
                    token_flippedPredictions_df = pd.concat(token_flippedPredictions, ignore_index=True)
                except:
                    token_flippedPredictions_df = pd.DataFrame(token_flippedPredictions)

                token_explanation = aggregateRankings(token_rankings, lenTriangles=len(allTriangles),
                                                maxLenAttributeSet=1000)

                if len(token_explanation) > 0:
                    token_sorted_attr_pairs = token_explanation.sort_values(ascending=False)
                    token_explanations = token_sorted_attr_pairs.loc[token_sorted_attr_pairs.values == token_sorted_attr_pairs.values.max()]
                    token_filtered = [i for i in token_explanations.keys() if
                                not any(all(c in i for c in b) and len(b) < len(i) for b in explanations.keys())]
                    for te in token_filtered:
                        token_filtered_exp[te] = token_explanations[te]

            return token_filtered_exp, token_flippedPredictions_df, allTriangles


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

def createTokenPerturbationsFromTriangle(triangleIds, sourcesMap, attributes, maxLenAttributeSet, classToExplain,
                                    lprefix='ltable_', rprefix='rtable_', use_tokens: bool = False):
    # generate power set of attributes
    allAttributesSubsets = list(_powerset(attributes, 1, 1))
    triangle = __getRecords(sourcesMap, triangleIds[:3])  # get triangle values
    perturbations = []
    perturbedAttributes = []
    diffs = []
    for subset in allAttributesSubsets:  # iterate over the attribute power set
        if classToExplain == 1:
            for i in range(1, 10):
                newRecord1 = triangle[0].copy()  # copy the l1 tuple
                newRecord2 = triangle[0].copy()  # copy the l1 tuple
                for att in subset:
                    val = str(triangle[2][att])  # copy the value for the given attribute from l2 of no-match l2, r1 pair into l1
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
                #perturbations.append(newRecord1)  # append the new record
                #perturbations.append(newRecord2)  # append the new record
        else:
            for i in range(1, 10):
                newRecord1 = triangle[2].copy()  # copy the l2 tuple
                newRecord2 = triangle[2].copy()  # copy the l2 tuple
                for att in subset:
                    val = str(triangle[0][att])  # copy the value for the given attribute from l1 of no-match l1, r1 pair into l2
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
                #perturbations.append(newRecord1)  # append the new record
                #perturbations.append(newRecord2)  # append the new record
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
