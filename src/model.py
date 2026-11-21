import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class HybridReadabilityModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_ling_features: int = 22,
        num_outputs: int = 3,
        dropout_rate: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.num_ling_features = num_ling_features

        hidden_dim = self.config.hidden_size  # 768 para roberta-base y distilbert-base
        fused_dim = hidden_dim + num_ling_features

        self.use_layernorm = use_layernorm and (num_ling_features > 0)
        if self.use_layernorm:
            self.fusion_norm = nn.LayerNorm(fused_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(fused_dim, num_outputs)

    def forward(self, input_ids, attention_mask, ling_features=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Vector [CLS] / <s> de la última capa oculta (compatible con RoBERTa y DistilBERT)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if self.num_ling_features > 0 and ling_features is not None:
            fused = torch.cat([cls_embedding, ling_features], dim=-1)
            if self.use_layernorm:
                fused = self.fusion_norm(fused)
        else:
            fused = cls_embedding

        logits = self.classifier(self.dropout(fused))
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits