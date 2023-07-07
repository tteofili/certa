from langchain import PromptTemplate, HuggingFaceHub, OpenAI
import os, random
from certa.models.ermodel import ERModel
import numpy as np

os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
os.environ['OPENAI_API_KEY'] = ''

hf_model = 'tiiuae/falcon-7b-instruct'

def text_to_match(answer):
    no_match_score = 0
    match_score = 0
    if answer.lower().startswith("yes"):
        match_score = 1
    elif answer.lower().startswith("no"):
        no_match_score = 1
    else:
        if "yes".casefold() in answer.casefold():
            match_score = 1
        else:
            no_match_score = 1
    return no_match_score, match_score


class LLMERModel(ERModel):

    count = 0

    def __init__(self, model_type='hf', temperature=1, max_length=180, fake=False):
        template = "given the record:\n{ltuple}\n and the record:\n{rtuple}\n do they refer to the same entity in the real world?\nreply yes or no"
        super(LLMERModel, self).__init__()
        self.prompt = PromptTemplate(
            input_variables=["ltuple", "rtuple"],
            template=template,
        )
        self.fake = fake
        if model_type == 'hf':
            self.llm = HuggingFaceHub(repo_id=hf_model,
                                      model_kwargs={'temperature': temperature, 'max_length': max_length})
        elif model_type == 'openai':
            self.llm = OpenAI(temperature=temperature, max_length=max_length, model_name='gpt-3.5-turbo')

    def predict(self, x, mojito=False, **kwargs):
        self.count += 1
        xc = x.copy()
        if self.fake:
            if random.choice([True, False]):
                match_score = 1
                nomatch_score = 0
            else:
                match_score = 0
                nomatch_score = 1
        else:
            elt = []
            ert = []
            for c in x.columns:
                if c in ['ltable_id', 'rtable_id']:
                    continue
                if c.startswith('ltable_'):
                    elt.append(x[c])
                if c.startswith('rtable_'):
                    ert.append(x[c])
            question = self.prompt.format(ltuple=elt, rtuple=ert)
            answer = self.llm(question)
            print(answer)
            nomatch_score, match_score = text_to_match(answer)
        xc['nomatch_score'] = nomatch_score
        xc['match_score'] = match_score
        if mojito:
            full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
            xc = full_df
        print(self.count)
        return xc
