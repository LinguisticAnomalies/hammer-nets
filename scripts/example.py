"""
scripts for function testing
"""
import os
import math
import random
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
DEVICE = "cuda"
USE_GPU = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def model_driver(input_text, model, tokenizer):
    """
    get the output from the model

    :param input_text: the transcript from a row of dataset
    :type input_text: Pandas.Series
    :param model: the GPT-2 model for evaluation
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    """
    encodings = tokenizer(input_text, return_tensors="pt",
                          return_attention_mask=True,
                          add_special_tokens=True,
                          truncation=True,
                          max_length=1024)
    if USE_GPU:
        input_ids = encodings["input_ids"]
        att_mask = encodings["attention_mask"]
        input_ids = input_ids.to(DEVICE)
        att_mask = att_mask.to(DEVICE)
        outputs = model(input_ids, attention_mask=att_mask, labels=input_ids)
        return outputs
    else:
        input_ids = encodings["input_ids"]
        att_mask = encodings["attention_mask"]
        outputs = model(input_ids, attention_mask=att_mask, labels=input_ids)
        return outputs


def break_attn_heads_by_layer(model, share, layer):
    """
    set certain percentage attention heads to zero at specific layer
    return the modified model
    :param model: the oringal GPT-2 model to be modified
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param share: % of attention heads to be zeroed,
                  25%, 50%, and 100% are supported
    :type share: int
    :param layer: the specific layer to be modified,
                  ranging from 0 to 11
    :type layer: int

    :return: the modified model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    random.seed(42)
    torch.manual_seed(42)
    head_offsets = [1536, 1536+64, 1536+128, 1536+192, 1536+256,
                    1536+320, 1536+384, 1536+448, 1536+512,
                    1536+576, 1536+640, 1536+704]
    offset = int(64*(share/100))
    batch = 64
    # randomly assign certain share of attention heads to zero
    random_list = random.sample([1] * offset + [0]*(batch-offset), batch)
    # if in the current index, the random list is 1,
    # then change the corresponding attention head from the model to 0
    for head in head_offsets:
        for i in range(64):
            if random_list[i] == 1:
                for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                    model.transformer.h[layer].attn.c_attn.weight[row][head+i] = \
                        model.transformer.h[layer].attn.c_attn.weight[row][head+i].mul(0)
    return model


if __name__ == "__main__":
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")

    for i in [0, 1, 2, 3, 4, 8, 10]:
        model_dem = break_attn_heads_by_layer(model_dem, 50, i)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    train_frame = pd.read_csv("data/address_train.csv")
    model_con.eval()
    model_dem.eval()
    if USE_GPU:
        model_con.to(DEVICE)
        model_dem.to(DEVICE)
    for _, row in train_frame.iterrows():
        with torch.no_grad():
            trans = row["text"]
            con_outputs = model_driver(trans, model_con, gpt_tokenizer)
            con_perp = math.exp(con_outputs[0].item())
            dem_outputs = model_driver(trans, model_dem, gpt_tokenizer)
            dem_perp = math.exp(dem_outputs[0].item())
            print('{:.2f} {:7.2f}'.format(con_perp, dem_perp))