from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

# load dataset from disk
train_dataset = load_from_disk('conll2003_train_split')
label_list = train_dataset.features['ner_tags'].feature.names
num_labels = len(label_list)

print(train_dataset[0])

train_dataset = train_dataset.remove_columns(['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])
print(train_dataset[0])

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
data_collator = DataCollatorForTokenClassification(tokenizer)

# construct DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
print(next(iter(train_loader)))
