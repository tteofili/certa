import random

import numpy as np
import pandas as pd


def expl_eval(class_to_explain, explanation_attributes, e_score, lsource, l_record, model, prediction, rsource,
              r_record, predict_fn):
    results = []
    if len(explanation_attributes) > 0:
        for i in range(2):
            lt = l_record.copy()
            rt = r_record.copy()
            perturb = 'sample'
            for e in explanation_attributes:
                if len(e) > 0:
                    if i == 0:
                        perturb = 'drop'
                        lt[e] = ''
                        rt[e] = ''
                    else:
                        f_class = abs(class_to_explain - 1)
                        if 'class' in lsource.columns:
                            ls_flip = lsource[lsource['class'] == f_class]
                            rs_flip = rsource[rsource['class'] == f_class]
                        else:
                            ls_flip = lsource.copy()
                            rs_flip = rsource.copy()
                        perturb = 'sample'
                        randint = random.randint(0, len(ls_flip) - 1)
                        lt[e] = ls_flip.iloc[randint][e]
                        randint = random.randint(0, len(rs_flip) - 1)
                        rt[e] = rs_flip.iloc[randint][e]
            df = pd.DataFrame(lt.add_prefix('ltable_').append(rt.add_prefix('rtable_'))).transpose()
            df = df.drop([c for c in ['ltable_id', 'rtable_id'] if c in df.columns], axis=1)
            modified_tuple_prediction = predict_fn(df, model)[['nomatch_score', 'match_score']].values[0]
            modified_class = np.argmax(modified_tuple_prediction)
            flip = abs(modified_class - class_to_explain)
            class_probability = prediction[class_to_explain]
            drop = class_probability - modified_tuple_prediction[class_to_explain]
            n_drop = (class_probability - modified_tuple_prediction[class_to_explain]) / class_probability
            impact = float(flip or (modified_tuple_prediction[class_to_explain] < 0.5 * class_probability))
            n_impact = float(flip or (
                    modified_tuple_prediction[class_to_explain] / class_probability < 0.5 * class_probability))
            ids = str(l_record['id']) + '-' + str(r_record['id'])
            new_row = {'match': class_to_explain, 'e_score': e_score, 'drop': drop, 'n_drop': n_drop,
                       'perturb': perturb, 'flip': flip, 'impact': impact, 'n_impact': n_impact,
                       'attributes': explanation_attributes,
                       'e_size': len(explanation_attributes), 'prediction': class_probability, 'row': ids}
            results.append(new_row)
    return pd.DataFrame(results)
