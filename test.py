import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from train import evaluate_func
from model  import NERBaseModel, MyBERT
import logging

is_CRF = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name =  './MyBERT_CRF.pth'
print(device)

# configure logging to file
logging.basicConfig(
    filename="test.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a"
)

# load dataset from disk
test_dataset = load_from_disk('conll2003_test_split')
label_list = test_dataset.features['ner_tags'].feature.names
num_labels = len(label_list)

test_dataset = test_dataset.remove_columns(['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
data_collator = DataCollatorForTokenClassification(tokenizer)

# construct DataLoader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)

if 'MyBERT' in  model_name:
    model =  MyBERT(num_labels=num_labels, is_CRF=is_CRF)
else:
    model = NERBaseModel(num_labels=num_labels, is_CRF=is_CRF)

model.load_state_dict(torch.load(model_name))
model.to(device)

results = evaluate_func(model, test_loader, label_list, is_CRF=is_CRF)
print(f"Precision: {results['overall_precision']:.4f}, Recall: {results['overall_recall']:.4f}, F1: {results['overall_f1']:.4f}")
logging.info(f"Model Name:{model_name}, precision: {results['overall_precision']:.4f}, Recall: {results['overall_recall']:.4f}, F1: {results['overall_f1']:.4f}")