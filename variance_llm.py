from certa.local_explain import expand_copies, generate_subsequences
import pandas as pd
import os
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
os.environ['OPENAI_API_KEY'] = 'sk-'

lprefix = 'ltable_'
rprefix = 'rtable_'
perturbed_copies = True
temperature = 0.99
max_length = 512
model_type = 'hf'
hf_model = 'tiiuae/falcon-7b-instruct'  # other models: 'huggyllama/llama-7b', ...

dataset_name = 'beers'
datadir = '/home/tteofili/dev/cheapER/datasets/' + dataset_name
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')

results = []
template = "given the record:\n{ltuple}\n and the record:\n{rtuple}\n do they refer to the same entity in the real world?\nreply yes or no"

prompt = PromptTemplate(
    input_variables=["ltuple", "rtuple"],
    template=template,
)

if model_type == 'hf':
    llm = HuggingFaceHub(repo_id=hf_model, model_kwargs={'temperature': temperature, 'max_length': max_length})
elif model_type == 'openai':
    llm = OpenAI(temperature=temperature, max_length=max_length, model_name='gpt-3.5-turbo')
else:
    llm = None

for idx in range(len(test[:5])):
    rand_row = test.iloc[idx]
    lid = int(rand_row['ltable_id'])
    ltuple = lsource.iloc[lid]
    rid = int(rand_row['rtable_id'])
    rtuple = rsource.iloc[rid]
    rand_row.head()

    ets = [(ltuple.drop(['id']).values.astype(str), rtuple.drop(['id']).values.astype(str))]

    if perturbed_copies:
        expand_df, _, _ = expand_copies(lprefix, lsource, ltuple, rtuple, rprefix, rsource)
        expand_df.head()
        for ei in range(len(expand_df)):
            expanded_row = expand_df.iloc[ei].astype(str)
            elt = []
            ert = []
            for c in expand_df.columns:
                if c in ['ltable_id', 'rtable_id']:
                    continue
                if c.startswith(lprefix):
                    elt.append(expanded_row[c])
                if c.startswith(rprefix):
                    ert.append(expanded_row[c])
            ets.append((elt, ert))
    els, ers = generate_subsequences(pd.DataFrame(ltuple).T, pd.DataFrame(rtuple).T)
    for el, er in zip(els, ers):
        ets.append((els.drop(['id']).values.astype(str), ers.drop(['id']).values.astype(str)))

    for er in ets:
        r1 = er[0]
        r2 = er[1]
        question = prompt.format(ltuple=r1, rtuple=r2)
        print(f'{question}')
        answer = llm(question)
        print(f'{answer}\n')
        results.append((r1, r2, answer))

results_df = pd.DataFrame(columns=['left', 'right', 'answer'], data=results)
results_df.to_csv('results.csv')
