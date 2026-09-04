from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


class DownStreamer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DownStreamer, self).__init__()
        self.fc1 = nn.Linear(output_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        out = self.fc2(x)
        return out
    
    
class TiSpell_RoBERTa(nn.Module):
    def __init__(self, model_name, tokenizer, dropout: Optional[float] = None):
        super(TiSpell_RoBERTa, self).__init__()
        # MaskedLM backbone (openpecha/tibetan_RoBERTa_*); CausalLM also loads but
        # emits decoder warnings. Logits feed the dual corrector heads.
        config = AutoConfig.from_pretrained(model_name)
        if dropout is not None:
            config.hidden_dropout_prob = dropout
            config.attention_probs_dropout_prob = dropout
        self.roberta = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, attn_implementation="eager"
        )
        self.vocab_size = len(tokenizer)
        self.roberta.resize_token_embeddings(self.vocab_size)
        self.config = self.roberta.config
        # Bottleneck through backbone hidden size (not vocab×vocab).
        hidden = self.config.hidden_size
        self.character_corrector = DownStreamer(hidden, self.vocab_size)
        self.syllable_corrector = DownStreamer(hidden, self.vocab_size)
        # Copy gate: per-position scalar deciding generate vs. copy input.
        # Init bias high so training starts near "copy everything", which is
        # the no-op baseline - the model then learns where editing pays.
        self.copy_gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, -2.0)   # sigmoid(-2) ~ 0.12

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_c + logit_s

        # Mix the generated distribution with a point mass on the input token.
        enc = outputs.hidden_states[-1]                      # [B, T, H]
        alpha = torch.sigmoid(self.copy_gate(enc))           # [B, T, 1]
        p_vocab = torch.softmax(logit, dim=-1)
        p_copy = torch.zeros_like(p_vocab)
        p_copy.scatter_(2, input_ids.unsqueeze(-1), 1.0)
        p = alpha * p_vocab + (1.0 - alpha) * p_copy
        log_p = torch.log(p.clamp_min(1e-10))

        # log_p replaces logits: argmax is unchanged, but the loss must be
        # nll_loss rather than cross_entropy. alpha returned for logging.
        return log_p, alpha
    
class TiSpell_RoBERTa_wo_Res(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(TiSpell_RoBERTa_wo_Res, self).__init__()
        self.roberta = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        self.vocab_size = len(tokenizer)
        self.roberta.resize_token_embeddings(self.vocab_size)
        self.config = self.roberta.config
        self.character_corrector = DownStreamer(self.vocab_size, self.vocab_size)
        self.syllable_corrector = DownStreamer(self.vocab_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_s
        return logit, logit_c
    
class TiSpell_RoBERTa_wo_Pretrain(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(TiSpell_RoBERTa_wo_Pretrain, self).__init__()
        # self.roberta = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModelForCausalLM.from_config(config)
        self.vocab_size = len(tokenizer)
        self.roberta.resize_token_embeddings(self.vocab_size)
        self.config = self.roberta.config
        self.character_corrector = DownStreamer(self.vocab_size, self.vocab_size)
        self.syllable_corrector = DownStreamer(self.vocab_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_c + logit_s
        return logit, logit_c
    
class TiSpell_RoBERTa_FC1(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(TiSpell_RoBERTa_FC1, self).__init__()
        self.roberta = AutoModelForCausalLM.from_pretrained(model_name)
        self.roberta.resize_token_embeddings(len(tokenizer))
        self.vocab_size = len(tokenizer)
        self.character_corrector = nn.Linear(self.vocab_size, self.vocab_size)
        self.syllable_corrector = nn.Linear(self.vocab_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_c + logit_s
        return logit, logit_c
    

class DownStreamerH(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TiSpell_RoBERTa_CopyGate(nn.Module):
    def __init__(self, model_name, tokenizer, dropout=None):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        if dropout is not None:
            config.hidden_dropout_prob = dropout
            config.attention_probs_dropout_prob = dropout
        config.output_hidden_states = True
        self.roberta = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, attn_implementation="eager")
        self.vocab_size = len(tokenizer)
        self.roberta.resize_token_embeddings(self.vocab_size)
        self.config = self.roberta.config
        hidden = self.config.hidden_size
        self.character_corrector = DownStreamerH(hidden, self.vocab_size)
        self.syllable_corrector = DownStreamerH(hidden, self.vocab_size)
        self.copy_gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, -2.0)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        enc = outputs.hidden_states[-1]
        logit_c = self.character_corrector(enc)
        logit_s = self.syllable_corrector(enc)
        logit = logit_c + logit_s
        alpha = torch.sigmoid(self.copy_gate(enc))
        copy_bonus = torch.zeros_like(logit)
        copy_bonus.scatter_(2, input_ids.unsqueeze(-1), 1.0)
        logit = logit + (1.0 - alpha) * copy_bonus * 8.0
        return logit, logit_c
