import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer

def load_dataset(dataset_name: str, tokenizer, split='train') -> datasets.Dataset:
    dataset = datasets.load_dataset(dataset_name,
                                    split=split,
                                    trust_remote_code=True)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []

        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # each token's word id
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # set special tokens to -100
                if word_idx is None:
                    label_ids.append(-100)
                # set the first token to the label
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # set the following tokens to -100
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    tokenized_dataset.save_to_disk(f'./{dataset_name}_{split}_split')

    return tokenized_dataset

if __name__ == '__main__':
    # dataset = load_from_disk('conll2003_train_split')
    # print(dataset.features['ner_tags'].feature)

    dataset = load_dataset('eriktks/conll2003', AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True), split='test')
    # for i in range(5):
    #     example = dataset[i]
    #     print(example)

