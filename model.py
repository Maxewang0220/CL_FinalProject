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

            # 使用 CRF 层进行预测
            # 训练阶段：使用labels进行概率建模
            if labels is not None:
                # 训练阶段：返回负对数似然损失
                new_labels = labels.clone()
                new_labels[new_labels == -100] = 0  # 将 -100 替换为合法标签（例如0），以便 CRF 层计算
                loss = -self.crf(logits, labels=new_labels, mask=labels_mask)
                return loss
            else:
                # 预测阶段：使用 Viterbi 解码得到最优标签序列
                predictions = self.crf.viterbi_decode(logits, mask=labels_mask)
                return predictions

        return logits

class MyBERT(torch.nn.Module):
    def __init__(self, num_labels=18, embedding_dim=768, num_heads=12, num_layers=12, max_length=256):
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

    def forward(self, input_ids, attention_mask):
        # 获取输入序列长度
        seq_length = input_ids.size(1)
        
        # 词嵌入 + 位置嵌入
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings[:seq_length, :].unsqueeze(0)
        embeddings = word_embeddings + position_embeddings
        
        # 转换mask格式 [batch_size, seq_len] -> [seq_len, seq_len]
        attn_mask = self._create_attention_mask(attention_mask)
        
        # 通过所有Transformer层
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)
        
        # 最终分类层
        logits = self.classifier(hidden_states)
        return logits

    def _create_attention_mask(self, attention_mask):
        # 将padding mask转换为attention mask [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        extended_mask = attention_mask[:, None, None, :]
        return extended_mask.repeat(1, 1, attention_mask.size(-1), 1).float()

class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        
        # 多头注意力机制
        self.attention = torch.nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        
        # 前馈网络
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 4*embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*embedding_dim, embedding_dim)
        )
        
        # Layer norm
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        # === 修改点1：先进行层归一化 ===
        # Pre-LN结构：LayerNorm -> Attention -> Residual
        attn_output, _ = self.attention(
            query=self.norm1(x),  # 归一化在attention之前
            key=self.norm1(x),
            value=self.norm1(x),
            key_padding_mask=(attention_mask == 0).squeeze(1)
        )
        x = x + self.dropout(attn_output)  # 残差连接原始输入
        
        # === 修改点2：前馈网络也使用Pre-LN ===
        # Pre-LN结构：LayerNorm -> FFN -> Residual
        ffn_output = self.ffn(self.norm2(x))  # 归一化在FFN之前
        x = x + self.dropout(ffn_output)  # 残差连接原始输入
        
        return x