"""Evaluate confidence measure."""
import json
import operator
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from models.ermodel import ERModel
from models.utils import from_type


def get_faithfullness(model: ERModel, base_dir: str):
    np.random.seed(1)
    saliency_names = ['certa', 'landmark', 'mojito_c', 'mojito_d', 'shap']
    all_y = []
    np.random.seed(0)

    model_scores = []
    thresholds = [0.1, 0.2, 0.33, 0.5, 0.7, 0.9]

    examples_df = pd.read_csv(os.path.join(base_dir, 'examples.csv')).drop(['match', 'Unnamed: 0'], axis=1)
    attr_len = len(examples_df.columns) - 2
    aucs = []
    for saliency in saliency_names:
        print(saliency)

        saliency_df = pd.read_csv(os.path.join(base_dir, saliency + '.csv'))
        for threshold in thresholds:
            top_k = int(threshold * attr_len)
            test_set_df = examples_df.copy()
            for i in range(len(saliency_df)):
                explanation = saliency_df.iloc[i]['explanation']
                attributes_dict = json.loads(explanation.replace("'", "\""))
                sorted_attributes_dict = sorted(attributes_dict.items(), key=operator.itemgetter(1))
                top_k_attributes = sorted_attributes_dict[:top_k]
                for t in top_k_attributes:
                    test_set_df.at[i, t[0]] = ''
            evaluation = model.evaluation(test_set_df)
            model_scores.append(evaluation[2])
        print(thresholds, model_scores)
        auc_sal = auc(thresholds, model_scores)
        aucs.append(auc_sal)
        print(f'auc:{auc_sal} for {saliency}')



if __name__ == "__main__":
    model_type = 'dm'
    model = from_type('%s' % model_type)
    dataset = 'fodo_zaga'
    base_dir = '/home/tteofili/dev/certa/'
    model.load('%smodels/%s/%s' % (base_dir, model_type, dataset))
    faithfullness = get_faithfullness(model, '%squantitative/%s/%s' % (base_dir, dataset, model_type))
    print(faithfullness)
