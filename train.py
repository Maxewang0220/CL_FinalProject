import torch
import wandb
import evaluate
from jupyterlab.semver import valid
from torch.utils.data import DataLoader
from datasets import load_from_disk
from model import NERBaseModel, MyBERT
from transformers import AutoTokenizer, DataCollatorForTokenClassification


# define evaluation function
def evaluate_func(model, dataloader, label_list):
    # use seqeval to evaluate NER
    metric = evaluate.load("seqeval")

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Shape：(batch_size, seq_length)
            predictions = torch.argmax(logits, dim=2)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            # filter paddings
            for prediction, label in zip(predictions, labels):
                true_predictions = []
                true_labels = []
                for p, l in zip(prediction, label):
                    if l != -100:
                        # seqeval only accepts string labels
                        true_predictions.append(label_list[p])
                        true_labels.append(label_list[l])

                all_predictions.append(true_predictions)
                all_labels.append(true_labels)

    results = metric.compute(predictions=all_predictions, references=all_labels)

    model.train()

    return results


if __name__ == '__main__':
    # training hyperparameters
    lr = 2e-4
    weight_decay = 0.01
    num_epochs = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset from disk
    train_dataset = load_from_disk('conll2003_train_split')
    valid_dataset = load_from_disk('conll2003_valid_split')
    label_list = train_dataset.features['ner_tags'].feature.names
    num_labels = len(label_list)

    train_dataset = train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    valid_dataset = valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

    # baseline model
    model = NERBaseModel(num_labels=num_labels)
    model.to(device)

    # filter the frozen parameters
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                  weight_decay=weight_decay)
    # cross-entropy loss ignore paddings with label -100
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Initialize wandb
    wandb.init(project="CL_FinalProject", name="dependency-parsing", resume=False, config={
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
    })

    # training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_idx = 1

        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

            avg_loss = total_loss / batch_idx
            batch_idx += 1

            wandb.log({"batch_loss": loss.item(), "avg_loss": avg_loss})

            if i % 100 == 0:
                # verify on validation set
                results = evaluate_func(model, valid_loader, label_list)
                print(
                    f"Precision: {results['overall_precision']:.4f}, Recall: {results['overall_recall']:.4f}, F1: {results['overall_f1']:.4f}")

                wandb.log({"precision": results['overall_precision'], "recall": results['overall_recall'],
                           "f1": results['overall_f1']})

    wandb.finish()

    # save model
    torch.save(model.state_dict(), 'ner_base_model.pth')
