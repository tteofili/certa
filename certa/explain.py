import logging

import numpy as np
import pandas as pd

from certa import local_explain, triangles_method


def explain(l_tuple, r_tuple, lsource, rsource, predict_fn, dataset_dir, left=True, right=True, attr_length=-1,
            num_triangles: int = 100, token_parts: bool = True, lprefix='ltable_', rprefix='rtable_',
            max_predict: int = -1, generate_perturb=True):
    pc = np.argmax(local_explain.get_original_prediction(l_tuple, r_tuple, predict_fn))
    print('local samples')
    local_samples, gleft_df, gright_df = local_explain.dataset_local(l_tuple, r_tuple, lsource, rsource,
                                                                     predict_fn, lprefix, rprefix,
                                                                     class_to_explain=pc,
                                                                     datadir=dataset_dir,
                                                                     use_w=right, use_y=left,
                                                                     num_triangles=num_triangles,
                                                                     token_parts=token_parts,
                                                                     max_predict=max_predict,
                                                                     generate_perturb=generate_perturb)

    if attr_length <= 0:
        attr_length = min(len(l_tuple) - 1, len(r_tuple) - 1)
    if len(local_samples) > 0:
        extended_sources = [pd.concat([lsource, gright_df]), pd.concat([rsource, gleft_df])]
        pns, pss, cf_ex, triangles = triangles_method.explain_samples(local_samples, extended_sources, predict_fn,
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
