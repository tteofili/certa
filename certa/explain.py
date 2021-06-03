import numpy as np
import pandas as pd

from certa import local_explain, triangles_method
from certa.local_explain import get_original_prediction


def explain(l_tuple, r_tuple, lsource, rsource, predict_fn, dataset_dir, fast: bool = True):
    predicted_class = np.argmax(get_original_prediction(l_tuple, r_tuple, predict_fn))
    local_samples, gleft_df, gright_df = local_explain.dataset_local(l_tuple, r_tuple, lsource, rsource,
                                                                     predict_fn, class_to_explain=predicted_class,
                                                                     datadir=dataset_dir, use_predict=not fast)

    explanation, _, _ = triangles_method.explainSamples(local_samples, [pd.concat([lsource, gright_df]),
                                                                        pd.concat([rsource, gleft_df])],
                                                        predict_fn, predicted_class, attribute_combine=True,
                                                        check=False, discard_bad=False)
    return explanation
