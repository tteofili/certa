import numpy as np
import pandas as pd

from certa import local_explain, triangles_method
from certa.local_explain import get_original_prediction


def explain(l_tuple, r_tuple, lsource, rsource, predict_fn, dataset_dir, fast: bool = True, left=True, right=True,
            attr_length=-1, check=False, discard_bad=False, num_triangles=100, return_top=False, token_parts=False,
            contrastive: bool = False):
    predicted_class = np.argmax(get_original_prediction(l_tuple, r_tuple, predict_fn))
    local_samples, gleft_df, gright_df = local_explain.dataset_local(l_tuple, r_tuple, lsource, rsource,
                                                                     predict_fn, class_to_explain=predicted_class,
                                                                     datadir=dataset_dir, use_predict=not fast,
                                                                     use_w=right, use_y=left,
                                                                     num_triangles=num_triangles, token_parts=token_parts)

    if attr_length <= 0:
        attr_length = min(len(l_tuple) - 2, len(r_tuple) - 2)

    explanation, flipped, triangles = triangles_method.explainSamples(local_samples, [pd.concat([lsource, gright_df]),
                                                                                      pd.concat([rsource, gleft_df])],
                                                                      predict_fn, predicted_class,
                                                                      check=check, discard_bad=discard_bad,
                                                                      attr_length=attr_length,
                                                                      return_top=return_top)
    if predicted_class == 0:
        explanation = explanation.apply(lambda x: x * -1)

    if contrastive:
        cf_class = abs(1 - predicted_class)
        cf_local_samples, cf_gleft_df, cf_gright_df = local_explain.dataset_local(l_tuple, r_tuple, lsource, rsource,
                                                                         predict_fn, class_to_explain=cf_class,
                                                                         datadir=dataset_dir, use_predict=not fast,
                                                                         use_w=right, use_y=left,
                                                                         num_triangles=num_triangles,
                                                                         token_parts=token_parts)

        if len(local_samples) > 0:
            cf_explanation, cf_flipped, cf_triangles = triangles_method.explainSamples(cf_local_samples,
                                                                                       [pd.concat([lsource, cf_gright_df]),
                                                                               pd.concat([rsource, cf_gleft_df])],
                                                                                       predict_fn, cf_class,
                                                                                       check=check, discard_bad=discard_bad,
                                                                                       attr_length=attr_length,
                                                                                       return_top=return_top, contrastive=True)
            if predicted_class == 1:
                cf_explanation = cf_explanation.apply(lambda x: x * -1)

            explanation = explanation + cf_explanation
            flipped = pd.concat([flipped, cf_flipped], axis=0)
            triangles = triangles + cf_triangles


    return explanation, flipped, triangles
