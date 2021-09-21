"""Evaluate confidence measure."""
import argparse
import json
import os
import random
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler


def sample(X, y, mode='up'):
    buckets_idx = defaultdict(lambda: [])
    buckets_size = defaultdict(lambda: 0)
    for i, _y in enumerate(y):
        buckets_size[int(_y * 10)] += 1
        buckets_idx[int(_y * 10)].append(i)

    if mode == 'up':
        sample_size = max(list(buckets_size.values()))

    if mode == 'down':
        sample_size = min(list(buckets_size.values()))

    if mode == 'mid':
        sample_size = (max(list(buckets_size.values())) - min(
            list(buckets_size.values()))) // 2

    new_idx = []

    for _, bucket_ids in buckets_idx.items():
        do_replace = True
        if sample_size <= len(bucket_ids):
            do_replace = False
        chosen = np.random.choice(bucket_ids, sample_size, replace=do_replace)
        new_idx += chosen.tolist()

    random.shuffle(new_idx)

    return X[new_idx], y[new_idx]


def get_confidence(base_dir: str):
    np.random.seed(1)
    saliency_names = ['certa', 'landmark', 'mojito_c', 'mojito_d', 'shap']
    all_y = []
    examples_df = pd.read_csv(os.path.join(base_dir, 'examples.csv'))
    for saliency in saliency_names:
        print(saliency)
        test_scores = []

        print(base_dir)

        # full_model_path = os.path.join(args.models_dir, model_path)
        # predictsions_path = full_model_path + '.predictions'
        saliency_path = os.path.join(base_dir,
                                     f"{saliency}.csv")

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
            instance_logits = np.fromstring(logits[i].replace('[','').replace(']',''), dtype=float, sep=' ')

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
            pred = reg.predict(X_train)
            test_pred = reg.predict(X_test)

            all_metrics = []
            for metric in [mean_absolute_error, max_error]:
                all_metrics.append(metric(y_test, test_pred))
            scores.append(all_metrics)
            coefs.append(reg.coef_)

            test_scores.append([np.mean([_s[i] for _s in scores]) for i in
                            range(len(scores[0]))])

        print(' '.join([f"{np.mean([_s[l] for _s in test_scores]):.3f} "
                    f"($\pm$ {np.std([_s[l] for _s in test_scores]):.3f})"
                    for l in range(len(test_scores[0]))]), flush=True)



if __name__ == "__main__":
    get_confidence('/home/tteofili/dev/certa/quantitative/beers/deeper')