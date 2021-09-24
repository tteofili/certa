from models.DeepER import DeepERModel
from models.bert import EMTERModel
from models.dm import DMERModel
from models.ermodel import ERModel


def from_type(type: str):
    model = ERModel()
    if "dm" == type:
        model = DMERModel()
    elif "deeper" == type:
        model = DeepERModel()
    elif "emt" == type:
        model = EMTERModel()
    elif "ditto" == type:
        model = None
    return model