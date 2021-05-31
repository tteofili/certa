from eval_deeper import eval_deeper
from eval_dm import eval_dm
from eval_emt import eval_emt

samples = 10
filtered_datasets = ['abt_buy', 'dirty_dblp_scholar', 'dirty_amazon_itunes', 'amazon_google',
                     'dirty_walmart_amazon', 'dirty_dblp_acm', 'itunes_amazon', 'fodo_zaga',
                     'dblp_acm', 'dblp_scholar', 'itunes_amazon', 'walmart_amazon']
evals_list_deeper = eval_deeper(samples=samples, filtered_datasets=filtered_datasets)
evals_list_dm = eval_dm(samples=samples, filtered_datasets=filtered_datasets)
evals_list_emt = eval_emt(samples=samples, filtered_datasets=filtered_datasets)