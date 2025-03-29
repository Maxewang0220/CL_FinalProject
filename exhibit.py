import torch
from datasets import load_from_disk
from model import NERBaseModel, MyBERT
from transformers import AutoTokenizer

is_CRF = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = './BERT_BASE.pth'
print(device)

raw_test_dataset = load_from_disk('conll2003_test_split')
raw_test_dataset = raw_test_dataset.remove_columns(['id', 'pos_tags', 'chunk_tags'])
label_list = raw_test_dataset.features['ner_tags'].feature.names
num_labels = len(label_list)

if 'MyBERT' in model_name:
    model = MyBERT(num_labels=num_labels, is_CRF=is_CRF)
else:
    model = NERBaseModel(num_labels=num_labels, is_CRF=is_CRF)

model.load_state_dict(torch.load(model_name))

model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True)

num_examples_to_show = 20
shown = 0

# traverse the raw dataset, tokenize the raw text to get input_ids, then feed into the model for prediction
for example in raw_test_dataset:
    tokens = example['tokens']  # original words list

    # tokenize the words and get the corresponding token ids
    encoding = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, truncation=True)
    input_ids = encoding['input_ids']

    input_ids_tensor = torch.tensor([input_ids]).to(device)
    attention_mask = torch.tensor([encoding['attention_mask']]).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
    # CRF outputs a list of predicted label sequences
    if is_CRF:
        pred_tags_ids = outputs[0]  # list of list
    else:
        logits = outputs
        pred_tags_ids = torch.argmax(logits, dim=2)[0]

    print(pred_tags_ids)

    # convert the predicted label ids to label names
    word_ids = encoding.word_ids()
    pred_tags = []
    previous_word_idx = None

    print(f"Example {shown + 1}")
    print(f"{'Word':<15}{'NER tag':<15}{'Label':<15}")

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # only take the label when the current token corresponds to a new word
        if word_idx != previous_word_idx:
            previous_word_idx = word_idx
            word = example["tokens"][word_idx]
            NER_tag = label_list[pred_tags_ids[idx]]
            label = label_list[example["ner_tags"][word_idx]]
            print(f"{word:<15}{NER_tag:<15}{label:<15}")

    shown += 1
    if shown >= num_examples_to_show:
        break
