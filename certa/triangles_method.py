import logging
from collections import defaultdict
from functools import partialmethod
from itertools import combinations

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import random

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


SML=10

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


def token_perturbations_from_triangle(triangleIds, sourcesMap, attributes, maxLenAttributeSet, classToExplain, lprefix,
                                      rprefix, idx=None, check=False, max_combs=10, predict_fn=None, min_t=0.45, max_t=0.55,
                                      summarizer=None):
    triangle = __getRecords(sourcesMap, triangleIds, lprefix, rprefix)  # get triangle values
    if classToExplain == 1:
        support = triangle[2].copy()
        free = triangle[0].copy()
    else:
        support = triangle[0].copy()
        free = triangle[2].copy()

    # generate power set of token-attributes
    allAttributesSubsets = list(_powerset(attributes, maxLenAttributeSet, maxLenAttributeSet))
    filtered_attribute_sets = []
    for att_set in allAttributesSubsets:
        good = True
        las = 0
        lp = -1
        while good and las < len(att_set):
            atc = att_set[las]
            clp = attributes.index(atc)
            good = clp > lp
            las += 1
        if good:
            filtered_attribute_sets.append(att_set)
    #filtered_attribute_sets = random.sample(filtered_attribute_sets, 10)
    perturbations = []
    perturbedAttributes = []
    droppedValues = []
    copiedValues = []

    for subset in filtered_attribute_sets:  # iterate over the attribute/token power set
        if not all(elem.split('__')[0] in free.index.to_list() for elem in subset):
            continue

        repls = [] # list of replacement attribute_token items
        aa = [] # list of affected attributes
        replacements = dict()
        for tbc in subset:  # iterate over the attribute:token items
            affected_attribute = tbc.split('__')[0]  # attribute to be affected
            aa.append(affected_attribute)
            if affected_attribute in support.index: # collect all possible tokens in the affected attribute to be used as replacements from the support record
                replacement_value = support[affected_attribute]
                replacement_tokens = replacement_value.split(' ')
                replacements[affected_attribute] = replacement_tokens
                for rt in replacement_tokens: # create attribute_token items for each replacement token
                    new_repl = '__'.join([affected_attribute, rt])
                    if not new_repl in repls:
                        repls.append(new_repl)

        all_rt_combs = list(_powerset(repls, maxLenAttributeSet, maxLenAttributeSet))
        filtered_combs = []
        for comb in all_rt_combs:
            naas = []
            for rt in comb:
                aspl = rt.split('__')[0]
                if aspl not in support.index:
                    continue
                naas.append(aspl)
            if aa == naas:
                filtered_combs.append(comb)

        #filtered_combs = random.sample(filtered_combs, min(max_combs, len(filtered_combs)))
        for comb in filtered_combs:
            naas = []
            for rt in comb:
                naas.append(rt.split('__')[0])
            newRecord = free.copy()
            dv = []
            cv = []
            affected_attributes = []
            ic = 0
            for tbc in subset:  # iterate over the attribute_token items
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
                    if predict_fn is not None:
                        conf = predict_fn(pd.DataFrame(newRecord).T)['match_score'].values[0]
                        if conf > min_t and conf < max_t:
                            good = False
                    else:
                        for c in newRecord.columns:
                            good = good and all(t in idx for t in nltk.bigrams(newRecord[c].astype(str).split(' ')))
                            if not good:
                                break
                if good:
                    droppedValues.append(dv)
                    copiedValues.append(cv)
                    perturbations.append(newRecord)
                    perturbedAttributes.append(subset)

    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r2 = triangle[1].copy()
    r2_copy = [r2] * len(perturbations_df)
    r2_df = pd.DataFrame(r2_copy, index=np.arange(len(perturbations)))
    allPerturbations = pd.DataFrame()
    if len(perturbations_df) > 0:
        if perturbations_df.columns[0].startswith(lprefix):
            allPerturbations = pd.concat([perturbations_df, r2_df], axis=1)
        else:
            allPerturbations = pd.concat([r2_df, perturbations_df], axis=1)
        allPerturbations = allPerturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    allPerturbations['alteredAttributes'] = perturbedAttributes
    allPerturbations['droppedValues'] = droppedValues
    allPerturbations['copiedValues'] = copiedValues

    return allPerturbations


def get_row_string(fr, pr):
    for col in ['ltable_id', 'id', 'rtable_id']:
        if col in fr:
            fr = fr.drop(col)
        if col in pr:
            pr = pr.drop(col)
    row = '\t'.join([' '.join(fr.astype(str).values), ' '.join(pr.astype(str).values), '0'])
    return row


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


def lattice_stratified_process(depth, allTriangles, attributes, class_to_explain, predict_fn, sourcesMap, lprefix, rprefix, num_threads=-1):
    all_good = False
    if num_threads != 1:
        perturbations = Parallel(n_jobs=num_threads)(delayed(token_perturbations_from_triangle)
                                                     (triangle, sourcesMap, attributes, depth, class_to_explain, lprefix,
                                                      rprefix)
                                                     for triangle in tqdm(allTriangles))
        ne_perturbations = []
        t_i = 0
        for tr in tqdm(allTriangles):
            curr_pert = perturbations[t_i]
            if len(curr_pert) > 0:
                curr_pert['triangle'] = ' '.join(tr)
                ne_perturbations.append(curr_pert)
            t_i += 1
        if len(ne_perturbations) == 0:
            perturbations_df = pd.DataFrame()
        else:
            perturbations_df = pd.concat(ne_perturbations)
    else:
        t_i = 0
        perturbations = []
        for triangle in tqdm(allTriangles):
            try:
                currentPerturbations = token_perturbations_from_triangle(triangle, sourcesMap, attributes, depth,
                                                                         class_to_explain, lprefix, rprefix)
                currentPerturbations['triangle'] = ' '.join(triangle)
                perturbations.append(currentPerturbations)
            except Exception as e:
                pass
            t_i += 1
        if len(perturbations) == 0:
            perturbations_df = pd.DataFrame()
        else:
            try:

                perturbations_df = pd.concat(perturbations)
            except:
                perturbations_df = pd.DataFrame(perturbations)

    if len(perturbations_df) == 0 or 'alteredAttributes' not in perturbations_df.columns:
        return perturbations_df, pd.DataFrame(), pd.DataFrame(), all_good, dict()
    currPerturbedAttr = perturbations_df.alteredAttributes.values

    predictions = predict_fn(
        perturbations_df.drop(['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle'], axis=1))
    predictions = pd.concat(
        [predictions, perturbations_df[['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle']]],
        axis=1)
    #all_predictions = pd.concat([all_predictions, predictions])
    proba = predictions[['nomatch_score', 'match_score']].values

    curr_flippedPredictions = predictions[proba[:, class_to_explain] < 0.5]

    #flippedPredictions.append(curr_flippedPredictions)

    ranking = getAttributeRanking(proba, currPerturbedAttr, class_to_explain)
    #rankings.append(ranking)

    if len(curr_flippedPredictions) == len(perturbations_df):
        logging.info(f'skipped predictions at depth >= {depth}')
        all_good = True
    else:
        logging.debug(f'predicted depth {depth}')

    return perturbations_df, predictions, curr_flippedPredictions, all_good, ranking


def perturb_predict_token(allTriangles, attributes, class_to_explain, predict_fn, sourcesMap, lprefix, rprefix, summarizer,
                          tf_idf_filter=True, num_threads=-1):

    if tf_idf_filter:
        t0 = __getRecords(sourcesMap, allTriangles[0], lprefix, rprefix)
        fr = t0[0]
        pr = t0[1]
        row_text = get_row_string(fr, pr)
        transformed_row_text = summarizer.transform(row_text.lower(), max_len=SML)
        filtered_attributes = []
        for ca in attributes:
            if ca.split('__')[1].lower() in transformed_row_text:
                filtered_attributes.append(ca)
        attributes = filtered_attributes

    #token_combinations = max(int(len(transformed_row_text.split(' '))), 3)
    token_combinations = 3

    if num_threads != 1:
        '''lattice_depth_results = Parallel(n_jobs=num_threads)(delayed(lattice_stratified_process)
                                                     (depth, allTriangles, attributes, class_to_explain, predict_fn, sourcesMap, lprefix, rprefix, num_threads)
                                                     for depth in range(token_combinations))'''
        #(perturbations_df, predictions, curr_flippedPredictions, all_good, ranking)
        perturbations_df, predictions, flippedPredictions, all_good, rankings = zip(*Parallel(n_jobs=1)(delayed(lattice_stratified_process)
                                                     (depth, allTriangles, attributes, class_to_explain, predict_fn, sourcesMap, lprefix, rprefix, num_threads)
                                                     for depth in range(token_combinations)))
        '''flippedPredictions_df = pd.concat(outputs_list[2])
        rankings = outputs_list[4]
        all_predictions = pd.concat(outputs_list[1])'''
        return pd.concat(flippedPredictions), rankings, pd.concat(predictions)
    else:
        all_predictions = pd.DataFrame()
        rankings = []
        flippedPredictions = []
        # lattice stratified predictions
        all_good = False
        for a in range(1, token_combinations):
            if all_good:
                break
            if num_threads != 1:
                perturbations = Parallel(n_jobs=num_threads)(delayed(token_perturbations_from_triangle)
                                                   (triangle, sourcesMap, attributes, a, class_to_explain, lprefix, rprefix)
                                                             for triangle in tqdm(allTriangles))
                t_i = 0
                for tr in tqdm(allTriangles):
                    perturbations[t_i]['triangle'] = ' '.join(tr)
                    t_i += 1
            else:
                t_i = 0
                perturbations = []
                for triangle in tqdm(allTriangles):
                    try:
                        currentPerturbations = token_perturbations_from_triangle(triangle, sourcesMap, attributes, a,
                                                                                 class_to_explain, lprefix, rprefix)
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
            if a < token_combinations and not all_good:
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
                logging.info(f'skipped predictions at depth >= {a}')
                all_good = True
            else:
                logging.debug(f'predicted depth {a}')
        try:
            flippedPredictions_df = pd.concat(flippedPredictions, ignore_index=True)
        except:
            flippedPredictions_df = pd.DataFrame(flippedPredictions)
        return flippedPredictions_df, rankings, all_predictions


def explain_samples(dataset: pd.DataFrame, sources: list, predict_fn: callable, lprefix, rprefix,
                    class_to_explain: int, attr_length: int, summarizer, check: bool = False,
                    discard_bad: bool = False, return_top: bool = False, persist_predictions: bool = False,
                    token: bool = False, two_step_token: bool = False, use_nec: bool = False):
    _renameColumnsWithPrefix(lprefix, sources[0])
    _renameColumnsWithPrefix(rprefix, sources[1])

    allTriangles, sourcesMap = getMixedTriangles(dataset, sources)
    if two_step_token:
        attributes = [col for col in list(sources[0]) if col not in [lprefix + 'id']]
        attributes += [col for col in list(sources[1]) if col not in [rprefix + 'id']]

        if len(allTriangles) > 0:
            attribute_ps, _, attribute_pn = attribute_level_expl(allTriangles, attr_length, attributes,
                                                                          check, class_to_explain, dataset,
                                                                          discard_bad, lprefix, persist_predictions,
                                                                          predict_fn, rprefix, sourcesMap)
            if use_nec:
                top_k_attr = 2
                sorted_pns = sorted(attribute_pn.items(), key=lambda kv: kv[1], reverse=True)
                topl = []
                topr = []
                spidx = 0
                while len(topl) < top_k_attr or len(topr) < top_k_attr:
                    c_attr = sorted_pns[spidx][0]
                    if c_attr.startswith(lprefix) and len(topl) < top_k_attr:
                        topl.append(c_attr)
                    if c_attr.startswith(rprefix) and len(topr) < top_k_attr:
                        topr.append(c_attr)
                    spidx += 1
                combs = topl + topr
            else:
                series = cf_summary(attribute_ps)
                combs = []
                for sc in series.index:
                    for scc in sc.split('/'):
                        combs.append(scc)
                combs = set(combs)

            record = pd.DataFrame(dataset.iloc[0]).T

            attributes = []
            for column in record.columns:
                if column not in ['label', 'id', lprefix + 'id', rprefix + 'id'] and column in combs:
                    tokens = str(record[column].values[0]).split(' ')
                    for t in tokens:
                        attributes.append(column + '__' + t)
            attr_length = len(attributes)

            if len(allTriangles) > 0:
                saliency, filtered_exp, flipped_predictions, allTriangles = token_level_expl(allTriangles, attr_length, attributes, class_to_explain, lprefix,
                                        persist_predictions, predict_fn, return_top, rprefix, sourcesMap, summarizer)
                return saliency, filtered_exp, flipped_predictions, allTriangles
            else:
                logging.warning(f'empty triangles !?')
                return dict(), pd.DataFrame(), pd.DataFrame(), []


    if token:
        record = pd.DataFrame(dataset.iloc[0]).T
        # we need to map records from series of attributes into series of tokens, attribute names are mapped to "original" token names
        attributes = []
        for column in record.columns:
            if column not in ['label', 'id', lprefix+'id', rprefix+'id']:
                tokens = str(record[column].values[0]).split(' ')
                for t in tokens:
                    attributes.append(column+'__'+t)
        attr_length = len(attributes)

        if len(allTriangles) > 0:
            return token_level_expl(allTriangles, attr_length, attributes, class_to_explain, lprefix,
                                    persist_predictions, predict_fn, return_top, rprefix, sourcesMap, summarizer)
        else:
            logging.warning(f'empty triangles !?')
            return dict(), pd.DataFrame(), pd.DataFrame(), []


    else:
        attributes = [col for col in list(sources[0]) if col not in [lprefix + 'id']]
        attributes += [col for col in list(sources[1]) if col not in [rprefix + 'id']]

        if len(allTriangles) > 0:
            explanation, flipped_predictions, saliency = attribute_level_expl(allTriangles, attr_length, attributes,
                                                                              check, class_to_explain, dataset,
                                                                              discard_bad, lprefix, persist_predictions,
                                                                              predict_fn, rprefix, sourcesMap)

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


def token_level_expl(allTriangles, attr_length, attributes, class_to_explain, lprefix, persist_predictions, predict_fn,
                     return_top, rprefix, sourcesMap, summarizer):
    flipped_predictions, rankings, all_predictions = perturb_predict_token(allTriangles, attributes,
                                                                           class_to_explain, predict_fn, sourcesMap,
                                                                           lprefix, rprefix, summarizer)
    if persist_predictions:
        all_predictions.to_csv('predictions.csv')
    explanation = aggregateRankings(rankings, lenTriangles=1, attr_length=attr_length)
    all_predictions['alteredAttributes'] = all_predictions['alteredAttributes'].astype(str).apply(
        lambda x: x.replace("'", '').replace('(', '').replace(',)', '').replace(', ', '/').replace(')', ''))
    perturb_count = all_predictions.groupby('alteredAttributes').size()
    for att in explanation.index:
        explanation[att] = explanation[att] / perturb_count[att]
    flips = len(flipped_predictions)
    saliency = dict()
    for ranking in rankings:
        for k, v in ranking.items():
            for a in k:
                if a not in saliency:
                    saliency[a] = 0
                if flips > 0:
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


def attribute_level_expl(allTriangles, attr_length, attributes, check, class_to_explain, dataset, discard_bad, lprefix,
                         persist_predictions, predict_fn, rprefix, sourcesMap):
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
            saliency[a] = len(allTriangles) / flips  # all attributes have a flip for the entire attribute set A
    for ranking in rankings:
        for k, v in ranking.items():
            for a in k:
                saliency[a] += v / flips
    return explanation, flipped_predictions, saliency


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
