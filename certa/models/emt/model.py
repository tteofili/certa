import os

from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer

def save_model(model, experiment_name, model_output_dir, epoch=None, tokenizer=None):
    if epoch:
        output_sub_dir = os.path.join(model_output_dir, experiment_name, "epoch_{}".format(epoch))
    else:
        output_sub_dir = os.path.join(model_output_dir, experiment_name)

    os.makedirs(output_sub_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_save.save_pretrained(output_sub_dir)

    if tokenizer:
        tokenizer.save_pretrained(output_sub_dir)

    return output_sub_dir


def load_model(model_dir, do_lower_case):
    config = DistilBertConfig.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, local_files_only=True, config=config)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case, local_files_only=True)
    if None == tokenizer:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=do_lower_case)

    return model, tokenizer
