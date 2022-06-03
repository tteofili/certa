CERTA
=======

Code for _CERTA_ (Computing ER explanations with TriAngles), an algorithm for computing saliency and counterfactual explanations for Entity Resolution models.

# Installation

To install _CERTA_ locally run :
```shell
pip install .
```

# Usage

Wrap the model whose predictions need to be explained using the [ERModel](models/ermodel.py) interface.
The _get_model_ utility method will load an existing model, if available, or train a new one using the data in the provided dataset.
E.g. for a _DeepMatcher_ model use:

```python
from certa.models.utils import get_model

model = get_model('dm', '/path/where/to/save', '/path/to/dataset', 'modelname')
```

Define a prediction function wrapping the _model.predict()_ method.

```python
def predict_fn(x, **kwargs):
    return model.predict(x, **kwargs)
```

Create a [CertaExplainer](certa/explain.py). 
_CERTA_ needs access to the data sources _lsource_ and _rsource_. 

```python
import pandas as pd
from certa.explain import CertaExplainer

lsource = pd.read_csv('/path/to/dataset/tableA.csv')
rsource = pd.read_csv('/path/to/dataset/tableB.csv')
certa_explainer = CertaExplainer(lsource, rsource)
```

To generate the prediction for the first two records in the data sources, do the following:

```python
import numpy as np
from certa.local_explain import get_original_prediction

l_tuple = lsource.iloc[0]
r_tuple = rsource.iloc[0]
prediction = get_original_prediction(l_tuple, r_tuple, predict_fn)
class_to_explain = np.argmax(prediction)
```

To explain the prediction using _CERTA_ :

```python
saliency, summary, cfs, triangles, lattices = certa_explainer.explain(l_tuple, r_tuple, predict_fn)
```
_CERTA_ returns:
* the saliency explanation within the _saliency_ pd.DataFrame 
* a _summary_ containing the set of attributes that has the highest probability of sufficiency of flipping the original prediction
* the generated counterfactual explanations within the _cfs_ pd.DataFrame 
* the list of open _triangles_ (in form of tuples of record ids) used to generate the explanations

# Examples

Examples of using _CERTA_ can be found in the following notebooks:
* [Explain DeepMatcher predictions](notebooks/sample.ipynb)
* [Explain Ditto predictions](https://gist.github.com/tteofili/b4c81a3de6aef40e8dfa27eaf22f116d)

# Citing CERTA

If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2203.12978):

```
@article{teofili2022effective,
  title={Effective Explanations for Entity Resolution Models},
  author={Teofili, Tommaso and Firmani, Donatella and Koudas, Nick and Martello, Vincenzo and Merialdo, Paolo and Srivastava, Divesh},
  journal={arXiv preprint arXiv:2203.12978},
  year={2022}
}
```
