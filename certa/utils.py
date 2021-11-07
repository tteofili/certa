import pandas as pd


def merge_sources(table, left_prefix, right_prefix, left_source, right_source, copy_from_table, ignore_from_table,
                  robust: bool = False):
    dataset = pd.DataFrame(columns={col: table[col].dtype for col in copy_from_table})
    ignore_column = copy_from_table + ignore_from_table

    for _, row in table.iterrows():
        leftid = row[left_prefix + 'id']
        rightid = row[right_prefix + 'id']

        new_row = {column: row[column] for column in copy_from_table}

        try:
            for id, source, prefix in [(leftid, left_source, left_prefix), (rightid, right_source, right_prefix)]:

                for column in source.keys():
                    if column not in ignore_column:
                        new_row[prefix + column] = source.loc[id][column]

            dataset = dataset.append(new_row, ignore_index=True)
        except:
            pass

        if robust:
            # symmetry
            sym_new_row = {column: row[column] for column in copy_from_table}
            try:
                for id, source, prefix in [(rightid, right_source, left_prefix), (leftid, left_source, right_prefix)]:

                    for column in source.keys():
                        if column not in ignore_column:
                            sym_new_row[prefix + column] = source.loc[id][column]

                dataset = dataset.append(sym_new_row, ignore_index=True)
            except:
                pass

            # identity
            lcopy_row = {column: row[column] for column in copy_from_table}
            try:
                for id, source, prefix in [(leftid, left_source, left_prefix), (leftid, left_source, right_prefix)]:

                    for column in source.keys():
                        if column not in ignore_column:
                            lcopy_row[prefix + column] = source.loc[id][column]

                lcopy_row['label'] = 1
                dataset = dataset.append(lcopy_row, ignore_index=True)
            except:
                pass

            rcopy_row = {column: row[column] for column in copy_from_table}
            try:
                for id, source, prefix in [(rightid, right_source, left_prefix), (rightid, right_source, right_prefix)]:

                    for column in source.keys():
                        if column not in ignore_column:
                            rcopy_row[prefix + column] = source.loc[id][column]

                rcopy_row['label'] = 1
                dataset = dataset.append(rcopy_row, ignore_index=True)
            except:
                pass
    return dataset


def diff(a: str, b: str):
    d = set(a.split(' ')).difference(b.split(' '))
    if len(d) == 0:
        d = '+' + str(set(b.split(' ')).difference(a.split(' ')))
    else:
        d = '-' + str(d)
    return d
