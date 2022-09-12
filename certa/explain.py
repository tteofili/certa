import logging

import numpy as np
import pandas as pd

from certa import local_explain, triangles_method
from certa.local_explain import generate_subsequences
from certa.utils import lattice, get_row


class CertaExplainer(object):

    def __init__(self, lsource, rsource, data_augmentation: str = 'on_demand'):
        '''
        Create the CERTA explainer
        :param lsource: the data source for "left" records
        :param rsource: the data source for "right" records
        :param data_augmentation: 'no' to avoid usage of DA at all, 'on_demand' to use it only when needed, 'always'
            to always use DA generated records even when the no. of found support records is sufficient.
        '''
        if data_augmentation in ['always', 'on_demand']:
            gen_left, gen_right = generate_subsequences(lsource, rsource)
            self.lsource = pd.concat([lsource, gen_left])
            self.rsource = pd.concat([rsource, gen_right])
            if data_augmentation == 'always':
                self.use_all = True
            else:
                self.use_all = False
        else:
            self.lsource = lsource
            self.rsource = rsource
            self.use_all = False

    def explain(self, l_tuple, r_tuple, predict_fn, left=True, right=True, attr_length=-1,
                num_triangles: int = 100, lprefix='ltable_', rprefix='rtable_',
                max_predict: int = -1, debug: bool = False):
        '''
        Explain the prediction generated by an ER model via its prediction function predict_fn on a pair of records
         l_tuple and r_tuple.
        :param l_tuple: the "left" record
        :param r_tuple: the "right" record
        :param predict_fn: the ER model prediction function
        :param left: whether to use left open triangles
        :param right: whether to use right open triangles
        :param attr_length: the maximum length of sets of attributes to be considered for generating an explanation
        :param num_triangles: number of open triangles to be used to generate the explanation
        :param lprefix: the prefix of attributes from the "left" table
        :param rprefix: the prefix of attributes from the "right" table
        :param max_predict: the maximum number of predictions to be performed by the ER model to generate the requested
        number of open triangles
        :param debug: whether to produce lattice data for debugging
        :return: saliency explanation, the probabilities of sufficiency, all the generated cf explanations, the open triangles
        '''
        prediction = local_explain.get_original_prediction(l_tuple, r_tuple, predict_fn)
        pc = np.argmax(prediction)
        support_samples, gleft_df, gright_df = local_explain.support_predictions(l_tuple, r_tuple, self.lsource,
                                                                                 self.rsource,
                                                                                 predict_fn, lprefix, rprefix,
                                                                                 class_to_explain=pc, use_w=left,
                                                                                 use_q=right, use_all=self.use_all,
                                                                                 num_triangles=num_triangles,
                                                                                 max_predict=max_predict)

        if attr_length <= 0:
            attr_length = min(len(l_tuple) - 1, len(r_tuple) - 1)
        if len(support_samples) > 0:
            extended_sources = [pd.concat([self.lsource, gright_df]), pd.concat([self.rsource, gleft_df])]
            pns, pss, cf_ex, triangles = triangles_method.explain_samples(support_samples, extended_sources, predict_fn,
                                                                          lprefix, rprefix, pc, attr_length=attr_length,
                                                                          persist_predictions=debug)
            cf_summary = triangles_method.cf_summary(pss)
            saliency_df = pd.DataFrame(data=[pns.values()], columns=pns.keys())
            if len(cf_ex) > 0:
                cf_ex['attr_count'] = cf_ex.alteredAttributes.astype(str) \
                    .str.split(',').str.len()
                cf_ex = cf_ex[cf_ex['alteredAttributes'].isin([tuple(k.split('/')) for k in cf_summary.keys()])] \
                    .astype(str).drop_duplicates(subset=['copiedValues', 'alteredAttributes', 'droppedValues'])
            lattices = []
            if debug:
                # generate lattice debug data
                triangle_predictions = pd.read_csv('predictions.csv')
                gbo = triangle_predictions.groupby('triangle')
                triangle_ids = list(gbo.groups.keys())
                for i in np.arange(len(triangle_ids)):
                    triangle = triangle_ids[i]
                    triangle_lattice = gbo.get_group(triangle)[['alteredAttributes', 'match_score']]
                    triangle_lattice['alteredAttributes'] = triangle_lattice['alteredAttributes'].apply(lambda x: tuple(
                        x.replace("'", '').replace('(', '').replace(')', '').replace(',', '').split(' ')))
                    lattice_dict = dict(zip(triangle_lattice.alteredAttributes, triangle_lattice.match_score))
                    triangle_edges = triangle.split(' ')
                    if triangle[0].startswith('0'):
                        powerset = [set()] + [set(s) for s in lattice_dict.keys()] + [
                            set([c for c in saliency_df.columns if c[0] == 'l'])]
                        if pc == 0:
                            f = extended_sources[0][extended_sources[0]['ltable_id'] == int(triangle_edges[2].split('@')[1])].iloc[0]
                            s = extended_sources[0][extended_sources[0]['ltable_id'] == int(triangle_edges[0].split('@')[1])].iloc[0]
                        else:
                            f = extended_sources[0][extended_sources[0]['ltable_id'] == int(triangle_edges[0].split('@')[1])].iloc[0]
                            s = extended_sources[0][extended_sources[0]['ltable_id'] == int(triangle_edges[2].split('@')[1])].iloc[0]
                        p = extended_sources[1][extended_sources[1]['rtable_id'] == int(triangle_edges[1].split('@')[1])].iloc[0]
                        tl_tuple = s
                        tr_tuple = p
                    else:
                        powerset = [set()] + [set(s) for s in lattice_dict.keys()] + [
                            set([c for c in saliency_df.columns if c[0] == 'r'])]
                        if pc == 0:
                            f = extended_sources[1][extended_sources[1]['rtable_id'] == int(triangle_edges[2].split('@')[1])].iloc[0]
                            s = extended_sources[1][extended_sources[1]['rtable_id'] == int(triangle_edges[0].split('@')[1])].iloc[0]
                        else:
                            f = extended_sources[1][extended_sources[1]['rtable_id'] == int(triangle_edges[0].split('@')[1])].iloc[0]
                            s = extended_sources[1][extended_sources[1]['rtable_id'] == int(triangle_edges[2].split('@')[1])].iloc[0]
                        p = extended_sources[0][extended_sources[0]['ltable_id'] == int(triangle_edges[1].split('@')[1])].iloc[0]
                        tl_tuple = p
                        tr_tuple = s

                    tl_tuple.index = tl_tuple.index.str.lstrip("ltable_")
                    tr_tuple.index = tr_tuple.index.str.lstrip("rtable_")

                    top_lattice_prediction = local_explain.get_original_prediction(tl_tuple, tr_tuple, predict_fn)
                    if np.argmax(top_lattice_prediction) == pc:
                        top_lattice_prediction = local_explain.get_original_prediction(tr_tuple, tl_tuple, predict_fn)
                    rank = [prediction[1]] + list(lattice_dict.values()) + [top_lattice_prediction[1]]
                    triangle_df = pd.concat([p, f, s], axis=1).T
                    triangle_df['type'] = ['pivot', 'free', 'support']
                    lattice_predictions = gbo.get_group(triangle).drop(
                        ['Unnamed: 0', 'triangle', 'droppedValues', 'copiedValues', 'nomatch_score'],
                        axis=1)

                    op = get_row(l_tuple, r_tuple).drop(['ltable_id', 'rtable_id'], axis=1)
                    op['alteredAttributes'] = ''
                    op['match_score'] = prediction[1]

                    sp = get_row(tl_tuple, tr_tuple)
                    if 'ltable_id' in sp.columns:
                        sp = sp.drop(['ltable_id'], axis=1)
                    if 'rtable_id' in sp.columns:
                        sp = sp.drop(['rtable_id'], axis=1)
                    sp['alteredAttributes'] = str(powerset[-1:][0])
                    sp['match_score'] = top_lattice_prediction[1]

                    lattice_predictions['alteredAttributes'] = lattice_predictions['alteredAttributes'].apply(
                        lambda x: tuple(
                            x.replace("'", '').replace('(', '').replace(')', '').replace(',', '').split(' ')))

                    lattice_predictions = pd.concat([sp, lattice_predictions, op], ignore_index=True)

                    lattice_predictions = lattice_predictions.sort_values(by="alteredAttributes",
                                                                          key=lambda x: x.str.count(
                                                                              '|'.join(['ltable_', 'rtable_'])),
                                                                          ascending=False)

                    latt = lattice(powerset, rank, triangle=lattice_predictions)
                    lattices.append(latt)

            return saliency_df, pss, cf_ex, triangles, lattices
        else:
            logging.warning('no triangles found -> empty explanation')
            return pd.DataFrame(), pd.Series(), pd.DataFrame(), [], []
