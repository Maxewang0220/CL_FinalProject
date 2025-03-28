import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from train import evaluate_func
from model import NERBaseModel, MyBERT
import logging


# 新增错误分析函数，统计各标签的错误比例及常见错误预测
def error_analysis(model, dataloader, label_list, device='cuda', is_CRF=False):
    # 用于统计每个标签的总出现次数
    total_counts = {label: 0 for label in label_list}
    # 用于统计错误预测，key 为真实标签，value 为字典：{预测标签: 次数}
    error_counts = {label: {} for label in label_list}

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if not is_CRF:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(logits, dim=2)
                predictions = predictions.cpu().numpy()
            else:
                predictions = model(input_ids=input_ids, attention_mask=attention_mask)

            labels = labels.cpu().numpy()

            # 遍历每个 token（跳过 padding 部分）
            for pred_seq, label_seq in zip(predictions, labels):
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    true_label = label_list[l]
                    pred_label = label_list[p]
                    total_counts[true_label] += 1
                    if true_label != pred_label:
                        error_counts[true_label][pred_label] = error_counts[true_label].get(pred_label, 0) + 1

    # 分析每个标签的错误比例及最常见的错误预测
    error_analysis_results = {}
    max_error_ratio = 0.0
    label_with_max_error = None
    for label in label_list:
        total = total_counts[label]
        errors = sum(error_counts[label].values())
        error_ratio = errors / total if total > 0 else 0
        most_common_error = None
        if error_counts[label]:
            most_common_error = max(error_counts[label].items(), key=lambda x: x[1])[0]
        error_analysis_results[label] = {
            "total": total,
            "errors": errors,
            "error_ratio": error_ratio,
            "most_common_error": most_common_error,
            "error_details": error_counts[label]
        }
        if error_ratio > max_error_ratio:
            max_error_ratio = error_ratio
            label_with_max_error = label

    model.train()
    return error_analysis_results, label_with_max_error, max_error_ratio


is_CRF = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = './MyBERT_CRF.pth'
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

if 'MyBERT' in model_name:
    model = MyBERT(num_labels=num_labels, is_CRF=is_CRF)
else:
    model = NERBaseModel(num_labels=num_labels, is_CRF=is_CRF)

model.load_state_dict(torch.load(model_name))
model.to(device)

results = evaluate_func(model, test_loader, label_list, is_CRF=is_CRF)
print(
    f"Precision: {results['overall_precision']:.4f}, Recall: {results['overall_recall']:.4f}, F1: {results['overall_f1']:.4f}")
logging.info(
    f"Model Name:{model_name}, precision: {results['overall_precision']:.4f}, Recall: {results['overall_recall']:.4f}, F1: {results['overall_f1']:.4f}")

error_analysis_results, label_with_max_error, max_error_ratio = error_analysis(model, test_loader, label_list,
                                                                               device=device, is_CRF=is_CRF)
print(f"Label with max error ratio: {label_with_max_error}, max error ratio: {max_error_ratio:.4f}")

for label, error_analysis_result in error_analysis_results.items():
    print(f"Label: {label}, Total: {error_analysis_result['total']}, Errors: {error_analysis_result['errors']}, "
          f"Error Ratio: {error_analysis_result['error_ratio']:.4f}, Most Common Error: {error_analysis_result['most_common_error']}")

    print(f"Error Details:{error_analysis_results['error_details']}")
