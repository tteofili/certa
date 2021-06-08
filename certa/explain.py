import numpy as np
import pandas as pd

from certa import local_explain, triangles_method
from certa.local_explain import get_original_prediction


def explain(l_tuple, r_tuple, lsource, rsource, predict_fn, dataset_dir, fast: bool = True, left=True, right=True,
            combinations=True, check=False, discard_bad=False, num_triangles=100, return_top=False):
    predicted_class = np.argmax(get_original_prediction(l_tuple, r_tuple, predict_fn))
    local_samples, gleft_df, gright_df = local_explain.dataset_local(l_tuple, r_tuple, lsource, rsource,
                                                                     predict_fn, class_to_explain=predicted_class,
                                                                     datadir=dataset_dir, use_predict=not fast,
                                                                     use_w=right, use_y=left,
                                                                     num_triangles=num_triangles)

    if combinations:
        maxLenAttributeSet = len(l_tuple) - 2
    else:
        maxLenAttributeSet = 1
    explanation, flipped, triangles = triangles_method.explainSamples(local_samples, [pd.concat([lsource, gright_df]),
                                                                        pd.concat([rsource, gleft_df])],
                                                        predict_fn, predicted_class,
                                                        check=check, discard_bad=discard_bad,
                                                        maxLenAttributeSet=maxLenAttributeSet, return_top=return_top)
    return explanation, flipped, triangles

