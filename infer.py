import torch
import re
from tqdm import tqdm
from option import parse_args
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.tispell_bert import *
from model.tispell_roberta import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 1. 加载 TTF 字体文件
font_path = None
font_prop = None

# 2. 设置字体为自定义字体
pass


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_path = "./weights/tispell_roberta"
    
    tokenizer = AutoTokenizer.from_pretrained('./weights/tispell_roberta', use_fast=False, do_lower_case=False)
    model = TiSpell_RoBERTa(model_path, tokenizer)
    model.load_state_dict(torch.load("./weights/tispell_roberta/tispell_roberta.pth", map_location=device))

    model.to(device).eval()

    text = ['སློབ་ཕྲུག་མང་པོས་ཉིན་ལྟར་སློབ་གྲྭར་ཕྱིན་ནས་ཡིག་ཚད་ལ་སྦྱོང་གོར་བྱེད་ཀྱི་ཡོད།']
    input_ids = tokenizer(text, truncation=True, padding="max_length", max_length=args.max_subword_length, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0], skip_special_tokens=True)
    layer_id = 11
    with torch.no_grad():
        logits, logits_c = model(**input_ids)
        '''
        attention = model.roberta(**input_ids).attentions[layer_id]
        for i in range(len(attention[0])):
            attention_head = attention[0, i].detach().cpu().numpy()
            attention_head = attention_head[1:len(tokens), 1:len(tokens)]
            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_head, xticklabels=tokens,
                        yticklabels=tokens, cmap='Blues', annot=False)
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            plt.savefig("images/layer_head_{}.png".format(str(i)), dpi=500)
        '''
        pred_ids = torch.argmax(logits, dim=-1)
        pred_mask = torch.argmax(logits_c, dim=-1)
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_masks = tokenizer.batch_decode(pred_mask, skip_special_tokens=False)
        for i in range(len(pred_texts)):
            pred_texts[i] = pred_texts[i].replace(' ', '')
            pred_masks[i] = pred_masks[i].replace(' ', '')
        print(input_ids['input_ids'])
        print(text)
        print(pred_texts)
        print(pred_masks)

if __name__ == "__main__":
    main()
