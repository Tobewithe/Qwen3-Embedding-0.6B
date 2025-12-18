import torch
from transformers import PreTrainedModel, AutoModel, PretrainedConfig,AutoConfig
class ClassificationConfig(PretrainedConfig):
    model_type = "classification_model"
    
    def __init__(self, num_labels=2, backbone_config_dict=None, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        # 关键：存储为 dict，确保可保存到 config.json
        self.backbone_config_dict = backbone_config_dict or {}
        
class ClassificationModel(PreTrainedModel):
    config_class = ClassificationConfig

    def __init__(self, config, backbone=None):
        super().__init__(config)
        
        # 关键：如果 backbone 已传入，直接用；否则留空（由 from_pretrained 填充）
        if backbone is not None:
            self.backbone = backbone
        else:
            if "model_type" in config.backbone_config_dict:
                _backbone_cfg_dict = config.backbone_config_dict.copy()
                _model_type = _backbone_cfg_dict.pop("model_type")
                backbone_config = AutoConfig.for_model(
                    _model_type, 
                    **_backbone_cfg_dict
                )
            else:
                backbone_config = PretrainedConfig.from_dict(config.backbone_config_dict)
            
            self.backbone = AutoModel.from_config(backbone_config, trust_remote_code=True)
        
        self.classifier = torch.nn.Linear(config.backbone_config_dict["hidden_size"], config.num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # 取最后一个有效 token
        seq_lengths = attention_mask.sum(dim=1)  # [batch]
        last_token_index = seq_lengths - 1      # 最后一个有效位置（0-based）
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        last_hidden = last_hidden_state[batch_indices, last_token_index]  # [batch, hidden_size]
        
        logits = self.classifier(last_hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}