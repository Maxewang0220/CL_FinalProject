import torch
from transformers import RobertaModel

# baseline model
class NERBaseModel(torch.nn.Module):
    def __init__(self, num_labels=18):
        super(NERBaseModel, self).__init__()

        # load pre-trained RoBERTa-base model
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # freeze RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False

        # project 768 hidden states to NER tags
        self.ffn = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Shape: (batch_size, seq_length, hidden_size:768)
        hidden_states = outputs.last_hidden_state

        # feed forward layer
        # Shape: (batch_size, seq_length, num_labels:18)
        logits = self.ffn(hidden_states)

        return logits

class MyBERT(torch.nn.Module):
    def __init__(self, num_labels=18):
        super(MyBERT, self).__init__()

    def forward(self, input_ids, attention_mask):
        pass