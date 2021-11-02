import logging

import numpy as np
import pandas as pd
import re

from certa import local_explain, triangles_method


def explain(l_tuple, r_tuple, lsource, rsource, predict_fn, dataset_dir, fast: bool = False, left=True, right=True,
            attr_length=-1, mode: str = 'open', num_triangles: int = 100, token_parts: bool = True,
            saliency: bool = True, lprefix='ltable_', rprefix='rtable_', max_predict: int = -1):
    predicted_class = np.argmax(local_explain.get_original_prediction(l_tuple, r_tuple, predict_fn))
    local_samples, gleft_df, gright_df = local_explain.dataset_local(l_tuple, r_tuple, lsource, rsource,
                                                                     predict_fn, lprefix, rprefix,
                                                                     class_to_explain=predicted_class,
                                                                     datadir=dataset_dir, use_predict=not fast,
                                                                     use_w=right, use_y=left,
                                                                     num_triangles=num_triangles,
                                                                     token_parts=token_parts,
                                                                     max_predict=max_predict)

    if attr_length <= 0:
        attr_length = min(len(l_tuple) - 1, len(r_tuple) - 1)
    if len(local_samples) > 0:
        explanations, counterfactual_examples, triangles = triangles_method.explainSamples(local_samples, [
            pd.concat([lsource, gright_df]),
            pd.concat([rsource, gleft_df])],
                                                                                           predict_fn, lprefix, rprefix,
                                                                                           predicted_class,
                                                                                           check=mode == 'closed',
                                                                                           discard_bad=mode == 'closed',
                                                                                           attr_length=attr_length,
                                                                                           return_top=False)

        cf_summary = triangles_method.cf_summary(explanations)

        saliency_df = pd.DataFrame()
        if saliency:
            sal = dict()

            alt_attrs = counterfactual_examples['alteredAttributes']
            for index, value in alt_attrs.items():
                attrs = re.sub("[\(\)',]", '', str(value)).split()
                if attrs[0].startswith('rprefix'):
                    attr_length = len(r_tuple) - 1
                else:
                    attr_length = len(l_tuple) - 1

                for attr in attrs:
                    nec_score = 1 / (pow(2, attr_length - 1))
                    if attr in sal:
                        ns = sal[attr] + nec_score
                    else:
                        ns = 2 * nec_score
                    sal[attr] = ns

            for k, v in sal.items():
                sal[k] = (v) / (len(triangles))

            saliency_df = pd.DataFrame(data=[sal.values()], columns=sal.keys())

        if len(counterfactual_examples) > 0:
            counterfactual_examples['attr_count'] = counterfactual_examples.alteredAttributes.astype(str)\
                .str.split(',').str.len()
            counterfactual_examples = counterfactual_examples.sort_values(by=['attr_count'])

        return saliency_df, cf_summary, counterfactual_examples, triangles
    else:
        logging.warning('no triangles found -> empty explanation')
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []
