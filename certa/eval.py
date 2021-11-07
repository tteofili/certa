import random

import numpy as np
import pandas as pd


def expl_eval(prediction, explanation_attributes, e_score, lsource, l_record, rsource,
              r_record, predict_fn, drop=True, sample=False):
    results = []
    class_to_explain = np.argmax(prediction)
    if len(explanation_attributes) > 0:
        if drop:
            try:
                lt = l_record.copy()
                rt = r_record.copy()
                perturb = 'drop'
                for e in explanation_attributes:
                    if len(e) > 0:
                        lt[e] = ''
                        rt[e] = ''
                df = pd.DataFrame(lt.add_prefix('ltable_').append(rt.add_prefix('rtable_'))).transpose()
                df = df.drop([c for c in ['ltable_id', 'rtable_id'] if c in df.columns], axis=1)
                modified_tuple_prediction = predict_fn(df)[['nomatch_score', 'match_score']].values[0]
                modified_class = np.argmax(modified_tuple_prediction)
                flip = abs(modified_class - class_to_explain)
                class_probability = prediction[class_to_explain]
                drop = class_probability - modified_tuple_prediction[class_to_explain]
                impact = int(flip or (modified_tuple_prediction[class_to_explain] < 0.5 * class_probability))
                ids = str(l_record['id']) + '-' + str(r_record['id'])
                new_row = {'match': class_to_explain, 'e_score': e_score, 'drop': drop,
                           'perturb': perturb, 'flip': flip, 'impact': impact,
                           'attributes': explanation_attributes,
                           'e_size': len(explanation_attributes), 'prediction': class_probability, 'row': ids}
                results.append(new_row)
            except:
                pass
        if sample:
            try:
                lt = l_record.copy()
                rt = r_record.copy()
                perturb = 'sample'
                for e in explanation_attributes:
                    if len(e) > 0:
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
                modified_tuple_prediction = predict_fn(df)[['nomatch_score', 'match_score']].values[0]
                modified_class = np.argmax(modified_tuple_prediction)
                flip = abs(modified_class - class_to_explain)
                class_probability = prediction[class_to_explain]
                drop = class_probability - modified_tuple_prediction[class_to_explain]
                impact = int(flip or (modified_tuple_prediction[class_to_explain] < 0.5 * class_probability))
                ids = str(l_record['id']) + '-' + str(r_record['id'])
                new_row = {'match': class_to_explain, 'e_score': e_score, 'drop': drop,
                           'perturb': perturb, 'flip': flip, 'impact': impact,
                           'attributes': explanation_attributes,
                           'e_size': len(explanation_attributes), 'prediction': class_probability, 'row': ids}
                results.append(new_row)
            except:
                pass

        for i in range(2):
            try:
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
                modified_tuple_prediction = predict_fn(df)[['nomatch_score', 'match_score']].values[0]
                modified_class = np.argmax(modified_tuple_prediction)
                flip = abs(modified_class - class_to_explain)
                class_probability = prediction[class_to_explain]
                drop = class_probability - modified_tuple_prediction[class_to_explain]
                impact = int(flip or (modified_tuple_prediction[class_to_explain] < 0.5 * class_probability))
                ids = str(l_record['id']) + '-' + str(r_record['id'])
                new_row = {'match': class_to_explain, 'e_score': e_score, 'drop': drop,
                           'perturb': perturb, 'flip': flip, 'impact': impact,
                           'attributes': explanation_attributes,
                           'e_size': len(explanation_attributes), 'prediction': class_probability, 'row': ids}
                results.append(new_row)
            except:
                pass
    return pd.DataFrame(results)


def eval_drop(prediction, explanation_attributes, l_record, r_record, predict_fn, lprefix='ltable',
              rprefix='rtable'):
    class_to_explain = np.argmax(prediction)
    drop = 0
    impact = 0
    if len(explanation_attributes) > 0:
        try:
            lt = l_record.copy()
            rt = r_record.copy()
            for e in explanation_attributes:
                if len(e) > 0 and e.startswith(lprefix):
                    lt[e] = ''
                if len(e) > 0 and e.startswith(rprefix):
                    rt[e] = ''
            df = pd.DataFrame(lt.append(rt)).transpose()
            df = df.drop([c for c in ['ltable_id', 'rtable_id'] if c in df.columns], axis=1)
            modified_tuple_prediction = predict_fn(df)[['nomatch_score', 'match_score']].values[0]
            modified_class = np.argmax(modified_tuple_prediction)
            flip = abs(modified_class - class_to_explain)
            class_probability = prediction[class_to_explain]
            drop = class_probability - modified_tuple_prediction[class_to_explain]
            impact = int(flip or (modified_tuple_prediction[class_to_explain] < 0.5 * class_probability))
        except:
            pass
    return drop, impact


def eval_sample(prediction, explanation_attributes, lsource, l_record, rsource, r_record, predict_fn):
    class_to_explain = np.argmax(prediction)
    drop = 0
    impact = 0
    try:
        lt = l_record.copy()
        rt = r_record.copy()
        for e in explanation_attributes:
            if len(e) > 0:
                f_class = abs(class_to_explain - 1)
                if 'class' in lsource.columns:
                    ls_flip = lsource[lsource['class'] == f_class]
                    rs_flip = rsource[rsource['class'] == f_class]
                else:
                    ls_flip = lsource.copy()
                    rs_flip = rsource.copy()
                randint = random.randint(0, len(ls_flip) - 1)
                lt[e] = ls_flip.iloc[randint][e]
                randint = random.randint(0, len(rs_flip) - 1)
                rt[e] = rs_flip.iloc[randint][e]
        df = pd.DataFrame(lt.add_prefix('ltable_').append(rt.add_prefix('rtable_'))).transpose()
        df = df.drop([c for c in ['ltable_id', 'rtable_id'] if c in df.columns], axis=1)
        modified_tuple_prediction = predict_fn(df)[['nomatch_score', 'match_score']].values[0]
        modified_class = np.argmax(modified_tuple_prediction)
        flip = abs(modified_class - class_to_explain)
        class_probability = prediction[class_to_explain]
        drop = class_probability - modified_tuple_prediction[class_to_explain]
        impact = int(flip or (modified_tuple_prediction[class_to_explain] < 0.5 * class_probability))
    except:
        pass
    return drop, impact
