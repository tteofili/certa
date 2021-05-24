#deeper
from eval_deeper import eval_deeper
from eval_dm import eval_dm
from eval_emt import eval_emt

filtered_datasets = ['dirty_dblp_scholar', 'dirty_amazon_itunes', 'dirty_walmart_amazon',
                                           'dirty_dblp_acm']
evals_list_deeper = eval_deeper(filtered_datasets=filtered_datasets)
evals_list_dm = eval_dm(filtered_datasets=filtered_datasets)
evals_list_emt = eval_emt(filtered_datasets=filtered_datasets)