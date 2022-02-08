import logging

import numpy as np
import pandas as pd

from certa import local_explain, triangles_method
from certa.local_explain import generate_subsequences


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
                max_predict: int = -1):
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
        :param use_all: whether to run the ER system on all available records
        :return: saliency explanation, cf explanation summary, all the generated cf explanations, the open triangles
        '''
        pc = np.argmax(local_explain.get_original_prediction(l_tuple, r_tuple, predict_fn))
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
                                                                          lprefix, rprefix, pc, attr_length=attr_length)
            cf_summary = triangles_method.cf_summary(pss)
            saliency_df = pd.DataFrame(data=[pns.values()], columns=pns.keys())
            if len(cf_ex) > 0:
                cf_ex['attr_count'] = cf_ex.alteredAttributes.astype(str) \
                    .str.split(',').str.len()
                cf_ex = cf_ex.sort_values(by=['attr_count'])
            return saliency_df, cf_summary, cf_ex, triangles
        else:
            logging.warning('no triangles found -> empty explanation')
            return pd.DataFrame(), pd.Series(), pd.DataFrame(), []
