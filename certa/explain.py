import logging

import numpy as np
import pandas as pd

from certa import local_explain, triangles_method
from certa.local_explain import generate_subsequences


class CertaExplainer(object):

    def __init__(self,lsource, rsource):
        gen_left, gen_right = generate_subsequences(lsource, rsource)
        self.lsource = pd.concat([lsource, gen_left])
        self.rsource = pd.concat([rsource, gen_right])

    def explain(self, l_tuple, r_tuple, predict_fn, left=True, right=True, attr_length=-1,
                num_triangles: int = 100, lprefix='ltable_', rprefix='rtable_',
                max_predict: int = -1):
        pc = np.argmax(local_explain.get_original_prediction(l_tuple, r_tuple, predict_fn))
        support_samples, gleft_df, gright_df = local_explain.support_predictions(l_tuple, r_tuple, self.lsource, self.rsource,
                                                                               predict_fn, lprefix, rprefix,
                                                                               class_to_explain=pc, use_w=left, use_q=right,
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
