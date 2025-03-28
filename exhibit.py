import torch
from datasets import load_from_disk
from model import  NERBaseModel, MyBERT
from transformers import AutoTokenizer

is_CRF = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name =  './BERT_BASE.pth'
print(device)

raw_test_dataset = load_from_disk('conll2003_test_split')
raw_test_dataset = raw_test_dataset.remove_columns(['id', 'pos_tags', 'chunk_tags'])
label_list = raw_test_dataset.features['ner_tags'].feature.names
num_labels = len(label_list)
# raw_test_dataset 中包含 'tokens' 和 'ner_tags'
# label_list 已经定义好

if 'MyBERT' in  model_name:
    model = MyBERT(num_labels=num_labels, is_CRF=is_CRF)
else:
    model = NERBaseModel(num_labels=num_labels, is_CRF=is_CRF)

model.load_state_dict(torch.load(model_name))

model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True)

num_examples_to_show = 20
shown = 0

# 遍历原始数据集，先对原始文本使用 tokenizer 得到 input_ids，然后送入模型预测（注意CRF情况）
for example in raw_test_dataset:
    tokens = example['tokens']  # 原始单词列表

    # 使用 fast tokenizer 进行 tokenization，并保持原始单词与tokenize后结果的对应（例如使用 is_split_into_words=True）
    encoding = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, truncation=True)
    input_ids = encoding['input_ids']
    
    # 构造 tensor，并移动到 device
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    # attention_mask 由 tokenizer 自动生成（注意形状为 [1, seq_len]）
    attention_mask = torch.tensor([encoding['attention_mask']]).to(device)

    # 获得模型预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
    # 如果使用 CRF，outputs 为预测的标签序列（列表形式）；否则为 logits，需要进行 argmax
    if is_CRF:
        pred_tags_ids = outputs[0]  # 取出预测的序列（假设返回的是 list of list）
    else:
        logits = outputs
        pred_tags_ids = torch.argmax(logits, dim=2)[0]
    
    print(pred_tags_ids)

    # 将预测的标签id转换为标签名称（注意可能需要考虑 tokenization带来的word-piece情况）
    # 这里我们通过 tokenizer.word_ids() 得到每个 token 对应的原始单词索引，然后只显示一个标签
    word_ids = encoding.word_ids()  # 长度与 input_ids 一致，可能含 None
    pred_tags = []
    previous_word_idx = None

    print(f"Example {shown+1}") 
    print(f"{'Word':<15}{'NER tag':<15}{'Label':<15}")

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # 仅在当前 token 对应的是一个新单词时取标签
        if word_idx != previous_word_idx:
            previous_word_idx = word_idx
            word = example["tokens"][word_idx]
            NER_tag = label_list[pred_tags_ids[idx]]
            label = label_list[example["ner_tags"][word_idx] ]
            print(f"{word:<15}{NER_tag:<15}{label:<15}")
    
    shown += 1
    if shown >= num_examples_to_show:
        break
