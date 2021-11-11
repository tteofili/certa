import math
import os
import traceback
import logging
import pandas as pd

from models.utils import from_type


def get_validity(model, rows_df, predicted_class):
    rowsc_df = rows_df.copy()
    if 'outcome' in rowsc_df.columns:
        return len(rowsc_df[rowsc_df[['outcome']].values == predicted_class]) / len(rowsc_df)
    else:
        if 'match_score' in rowsc_df.columns and 'nomatch_score' in rowsc_df.columns:
            predictions = rowsc_df
        else:
            predictions = model.predict(rowsc_df)
        proba = predictions[['nomatch_score', 'match_score']].values
        flipped_df = predictions[proba[:, predicted_class] < 0.5]
        return len(flipped_df) / len(rowsc_df)


def get_proximity(rows_df, original_row):
    proximity_all = 0
    for i in range(len(rows_df)):
        curr_row = rows_df.iloc[i]
        sum_cat_dist = 0
        if 'match_score' in curr_row:
            curr_row = curr_row.drop(
                ['alteredAttributes', 'match_score', 'nomatch_score', 'copiedValues', 'droppedValues', 'attr_count'])

        for c, v in curr_row.items():
            if c in curr_row and c in original_row and v == original_row[c]:
                sum_cat_dist += 1

        proximity = 1 - (1 / len(original_row)) * sum_cat_dist
        proximity_all += proximity
    return proximity_all / len(rows_df)


def get_diversity(expl_df):
    diversity = 0
    for i in range(len(expl_df)):
        for j in range(len(expl_df)):
            if i == j:
                continue
            curr_row1 = expl_df.iloc[i]
            curr_row2 = expl_df.iloc[j]
            sum_cat_dist = 0
            if 'match_score' in curr_row1:
                curr_row1 = curr_row1.drop(
                    ['alteredAttributes', 'match_score', 'nomatch_score', 'copiedValues', 'droppedValues',
                     'attr_count'])
            if 'match_score' in curr_row2:
                curr_row2 = curr_row2.drop(
                    ['alteredAttributes', 'match_score', 'nomatch_score', 'copiedValues', 'droppedValues',
                     'attr_count'])

            for c, v in curr_row1.items():
                if v != curr_row2[c]:
                    sum_cat_dist += 1

            dist = sum_cat_dist / len(curr_row1)
            diversity += dist
    return diversity / math.pow(len(expl_df), 2)


def get_sparsity(expl_df, instance):
    return 1 - get_proximity(expl_df, instance) / (len(expl_df.columns) / 2)


def cf_eval(model_type: str, samples=50, whitelist=[]):
    experiments_dir = 'cf/'
    base_dir = ''
    t = 10

    cf_dict = dict()
    for subdir, dirs, files in os.walk(experiments_dir):
        for dataset in dirs:
            if dataset not in whitelist:
                continue

            model = from_type('%s' % model_type)
            try:
                logging.info('loading model')
                model.load('%smodels/%s/%s' % (base_dir, model_type, dataset))

                base_cf_exp_dir = experiments_dir + dataset + '/' + model_type
                logging.info('loading cfs')
                examples_df = pd.read_csv(base_cf_exp_dir + '/examples.csv')

                cf_eval = dict()
                saliency_names = ['certa', 'dice_random', 'shapc', 'limec']
                for saliency in saliency_names:
                    logging.info(f'processing {saliency}')
                    validity = 0
                    proximity = 0
                    sparsity = 0
                    diversity = 0
                    length = 0
                    count = 0
                    for i in range(samples):
                        try:
                            # get cfs
                            expl_df = pd.read_csv(base_cf_exp_dir + '/' + str(i) + '/' + saliency + '.csv')

                            example_row = examples_df.iloc[i]
                            instance = example_row.drop(['ltable_id', 'rtable_id', 'match', 'label'])
                            score = example_row['match']
                            predicted_class = int(float(score) > 0.5)

                            # validity
                            validity += get_validity(model, expl_df[:t], predicted_class)

                            # proximity
                            proximity += get_proximity(expl_df[:t], instance)

                            # sparsity
                            sparsity += get_sparsity(expl_df[:t], instance)

                            # diversity
                            diversity += get_diversity(expl_df[:t])

                            length += len(expl_df)
                            count += 1
                        except:
                            pass
                    row = {'validity': validity / count, 'proximity': proximity / count,
                           'sparsity': sparsity / count, 'diversity': diversity / count,
                           'length': length / count}
                    logging.info(f'{saliency}:{row}')
                    cf_eval[saliency] = row

                print(f'{model_type}: cf-eval for {dataset}: {cf_eval}')
                cf_dict[dataset] = cf_eval
            except:
                print(traceback.format_exc())
                print(f'skipped {dataset}')
                pass
        return cf_dict


if __name__ == "__main__":
    samples = 20
    mtype = 'deeper'
    filtered_datasets = ['dirty_amazon_itunes', 'dirty_walmart_amazon', 'dirty_dblp_acm', 'dirty_dblp_scholar',
                         'fodo_zaga', 'beers', 'abt_buy',
                         'walmart_amazon', 'itunes_amazon', 'amazon_google',
                         'dblp_scholar', 'dblp_acm'
                         ]
    cf_dict = cf_eval(mtype, samples=samples, whitelist=filtered_datasets)
    print(cf_dict)