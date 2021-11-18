import math


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
