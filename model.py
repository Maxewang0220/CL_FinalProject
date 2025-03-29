import torch
from transformers import RobertaModel
from TorchCRF import CRF


# baseline model
class NERBaseModel(torch.nn.Module):
    def __init__(self, num_labels=18, is_CRF=False):
        super(NERBaseModel, self).__init__()

        # load pre-trained RoBERTa-base model
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # freeze RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False

        # project 768 hidden states to NER tags
        self.ffn = torch.nn.Linear(768, num_labels)

        self.is_CRF = is_CRF
        if self.is_CRF:
            # CRF layer
            self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Shape: (batch_size, seq_length, hidden_size:768)
        hidden_states = outputs.last_hidden_state

        # feed forward layer
        # Shape: (batch_size, seq_length, num_labels:18)
        logits = self.ffn(hidden_states)

        if self.is_CRF:
            # construct new labels mask
            labels_mask = attention_mask.bool() & (labels != -100)

            # use CRF layer for prediction
            # training stage: use labels for probability modeling
            if labels is not None:
                # training stage: return negative log likelihood loss
                new_labels = labels.clone()
                new_labels[new_labels == -100] = 0  # replace -100 with legal label 0 for CRF layer calculation
                loss = -self.crf(logits, labels=new_labels, mask=labels_mask)
                return loss
            else:
                # prediction stage: use Viterbi decoding to get the optimal label sequence
                predictions = self.crf.viterbi_decode(logits, mask=labels_mask)
                return predictions

        return logits


class MyBERT(torch.nn.Module):
    def __init__(self, num_labels=18, embedding_dim=768, num_heads=12, num_layers=12, max_length=256, is_CRF=False):
        super(MyBERT, self).__init__()

        # use pretrained RoBERTa tokenizer
        self.word_embeddings = torch.nn.Embedding(50265, embedding_dim)

        # position embedding 
        self.position_embeddings = torch.nn.Parameter(torch.randn(max_length, embedding_dim))

        # transfortmer block
        self.layers = torch.nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])

        # ffn classifer
        self.classifier = torch.nn.Linear(embedding_dim, num_labels)

        self.is_CRF = is_CRF
        if self.is_CRF:
            # CRF layer
            self.crf = CRF(num_labels)
            # froze all parameters except 'classifier' and 'crf'
            for name, param in self.named_parameters():
                if not (name.startswith("classifier") or name.startswith("crf")):
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        # acquire the length of input sequence
        seq_length = input_ids.size(1)

        # word embedding + position embedding
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings[:seq_length, :].unsqueeze(0)
        embeddings = word_embeddings + position_embeddings

        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        logits = self.classifier(hidden_states)

        if self.is_CRF:
            # construct new labels mask
            labels_mask = attention_mask.bool() & (labels != -100)

            if labels is not None:
                new_labels = labels.clone()
                new_labels[new_labels == -100] = 0
                loss = -self.crf(logits, labels=new_labels, mask=labels_mask)
                return loss
            else:
                predictions = self.crf.viterbi_decode(logits, mask=labels_mask)
                return predictions

        return logits

    def reinit_classifier(self):
        torch.nn.init.kaiming_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            torch.nn.init.zeros_(self.classifier.bias)


class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        # multi-head attention
        self.attention = torch.nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)

        # feed forward network
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 4 * embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * embedding_dim, embedding_dim)
        )

        # Layer norm
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        # Pre-LayerNorm: LayerNorm -> Attention -> Residual
        attn_output, _ = self.attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            key_padding_mask=(attention_mask == 0).squeeze(1)
        )
        x = x + self.dropout(attn_output)  # residual connection

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x
