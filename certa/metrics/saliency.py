import json
import operator
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc
from sklearn.metrics import max_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from certa.models.ermodel import ERModel


def get_confidence(saliency_names: list, base_dir: str):
    np.random.seed(1)
    all_y = []
    ci = dict()
    for saliency in saliency_names:
        print(saliency)
        test_scores = []

        saliency_df = pd.read_csv(os.path.join(base_dir, saliency + '.csv'))

        predictions = saliency_df['prediction']
        class_preds = predictions.apply(lambda x: np.argmax(x))
        logits = predictions.copy()
        saliencies = []
        classes = [0, 1]
        features = []
        y = []
        tokens = []

        for i in range(len(saliency_df)):
            instance_saliency = saliency_df.iloc[i]
            instance_sals = []
            instance_tokens = []
            for _cls in classes:
                cls_sals = []
                explanation = instance_saliency['explanation']
                attributes_dict = json.loads(explanation.replace("'", "\""))
                for _token, sal in attributes_dict.items():
                    if _cls == 0:
                        instance_tokens.append(_token)
                    if saliency == 'certa':
                        if _cls == class_preds[i]:
                            attr_sal = sal
                        else:
                            attr_sal = 1 - sal
                    else:
                        if _cls == 0:
                            if sal < 0:
                                attr_sal = abs(sal)
                            else:
                                attr_sal = 0
                        else:
                            if sal > 0:
                                attr_sal = sal
                            else:
                                attr_sal = 0
                    cls_sals.append(attr_sal)
                instance_sals.append(cls_sals)
            saliencies.append(instance_sals)
            tokens.append(instance_tokens)

        for i, instance in enumerate(saliencies):
            _cls = class_preds[i]
            instance_saliency = saliencies[i]
            instance_logits = np.fromstring(logits[i].replace('[', '').replace(']', ''), dtype=float, sep=' ')

            confidence_pred = instance_logits[_cls]
            saliency_pred = np.array(instance_saliency[_cls])

            left_classes = classes.copy()
            left_classes.remove(_cls)
            other_sals = [np.array(instance_saliency[c_]) for c_ in
                          left_classes]
            feats = []

            if len(classes) == 2:
                feats.append(sum(saliency_pred - other_sals[0]))
                feats.append(sum(saliency_pred - other_sals[0]))
                feats.append(sum(saliency_pred - other_sals[0]))

            else:
                feats.append(sum(np.max([saliency_pred - other_sals[0],
                                         saliency_pred - other_sals[1]],
                                        axis=0)))
                feats.append(sum(np.mean([saliency_pred - other_sals[0],
                                          saliency_pred - other_sals[1]],
                                         axis=0)))
                feats.append(sum(np.min([saliency_pred - other_sals[0],
                                         saliency_pred - other_sals[1]],
                                        axis=0)))

            y.append(confidence_pred)
            features.append(feats)

        features = MinMaxScaler().fit_transform(np.array(features))
        all_y += y
        y = np.array(y)

        rs = ShuffleSplit(n_splits=5, random_state=2)
        scores = []
        coefs = []
        for train_index, test_index in rs.split(features):
            X_train, y_train, X_test, y_test = features[train_index], y[
                train_index], features[test_index], y[test_index]
            reg = LinearRegression().fit(X_train, y_train)
            test_pred = reg.predict(X_test)

            all_metrics = []
            for metric in [mean_absolute_error, max_error]:
                all_metrics.append(metric(y_test, test_pred))
            scores.append(all_metrics)
            coefs.append(reg.coef_)

            test_scores.append([np.mean([_s[i] for _s in scores]) for i in
                                range(len(scores[0]))])

        ci_sal = dict()
        for l in range(len(test_scores[0])):
            d = dict()
            mu = np.mean([_s[l] for _s in test_scores])
            sigma = np.std([_s[l] for _s in test_scores])
            d['mean'] = mu
            d['std'] = sigma
            ci_sal[l] = d
        ci[saliency] = ci_sal

        print(' '.join([f"{np.mean([_s[l] for _s in test_scores]):.3f} "
                        f"($\pm$ {np.std([_s[l] for _s in test_scores]):.3f})"
                        for l in range(len(test_scores[0]))]), flush=True)
    return ci


def get_faithfulness(saliency_names: list, model: ERModel, base_dir: str, test_set_df: pd.DataFrame):
    np.random.seed(0)

    thresholds = [0.1, 0.2, 0.33, 0.5, 0.7, 0.9]

    attr_len = len(test_set_df.columns) - 2
    aucs = dict()
    for saliency in saliency_names:
        model_scores = []
        reverse = True
        saliency_df = pd.read_csv(os.path.join(base_dir, saliency + '.csv'))
        if saliency == 'certa':
            preds = saliency_df['match']

        for threshold in thresholds:
            top_k = int(threshold * attr_len)
            test_set_df_c = test_set_df.copy().astype(str)
            for i in range(len(saliency_df)):
                if int(preds[i]) == 0:
                    reverse = False
                explanation = saliency_df.iloc[i]['explanation']
                attributes_dict = json.loads(explanation.replace("'", "\""))
                if saliency == 'certa':
                    sorted_attributes_dict = sorted(attributes_dict.items(), key=operator.itemgetter(1),
                                                    reverse=True)
                else:
                    sorted_attributes_dict = sorted(attributes_dict.items(), key=operator.itemgetter(1),
                                                    reverse=reverse)
                top_k_attributes = sorted_attributes_dict[:top_k]
                for t in top_k_attributes:
                    test_set_df_c.at[i, t[0]] = ''
            evaluation = model.evaluation(test_set_df_c)
            model_scores.append(evaluation[2])
        auc_sal = auc(thresholds, model_scores)
        aucs[saliency] = auc_sal
    return aucs
