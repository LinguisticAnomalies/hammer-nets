"""
Utility functions
"""

import os
import re
import pickle
import math
import sys
import json
import pandas as pd
import numpy as np
import torch
from nltk.probability import FreqDist
from sklearn.metrics import roc_curve, auc


DEVICE = "cuda"
USE_GPU = True
head_offsets = [1536, 1536+64, 1536+128, 1536+192, 1536+256,
                1536+320, 1536+384, 1536+448, 1536+512,
                1536+576, 1536+640, 1536+704]
con_case = "okay the little boy is on a stool about to fall. the stool's about to upset. and he has a cookie in each hand handing about to hand one. and the water is running over into the dishpan there or into. and the mother or the lady is standing there drying a dish. two two cups and a plate are on the counter there. and and out the window there's a walkway and and. what's happening you said huh. okay that's that's what's happening i guess. thank"
dem_case = "well the little kid's falling off his stool. and the mother is having water run over the sink. well the water's running on the floor. under her feet. i'm looking outside but that yard is okay. the windows are open. the little girl is laughing at the boy falling off the chair. that that's bad."


def get_word_dist(input_frame, trans_col):
    """
    get the log word frequency distrubtion on the input frame, 
    :param pandas.DataFrame input_frame: the input frame with transcripts to evalute
    :param str trans_col: the column name represents transcript text
    :return: the log frequency for the transcript tokens
    :rtype: dict
    """
    word_dist = load_word_dist()
    log_lf = {}
    trans = input_frame[trans_col].values.tolist()
    for tran in trans:
        try:
            words = tran.split(" ")
            for word in set(words):
                log_lf[word.lower()] = get_word_lf(word, word_dist)
        except AttributeError:
            continue
    log_lf = sorted(log_lf.items(), key=lambda item: item[1])
    return log_lf


def get_word_lf(token, word_dist):
    """
    get the log lexical frequency for a specifc word
    :param str token: the word for calcualtion
    :param dict word_dict: the dictionary for the word raw frequency
    :return: the log lexical frequency for the word
    :rtype: float
    """
    return np.log(word_dist[token.lower()])


def load_word_dist():
    """
    load Subtlex.US.text file for estimating word frequency distribution
    save it to local file
    return the word frequency distribution
    :rtype: nltk.probability.FreqDist
    """
    if os.path.exists("word_dist.pkl"):
        with open("word_dist.pkl", "rb") as f:
            word_dist = pickle.load(f)
        return word_dist
    else:
        word_dist = FreqDist()
        sys.stdout.write("estimating frequency distribution...\n")
        with open("Subtlex.US.text", "r") as c:
            lines = c.readlines()
            for ln in lines:
                words = ln.split()
                for word in words:
                    word_dist[word.lower()] += 1
        sys.stdout.write("done\n")
        with open("word_dist.pkl", "wb") as f:
            pickle.dump(word_dist, f)
        return word_dist


def read_data(prefix_path, data_type):
    """
    read the data into DataFrame
    :param str prefix_path: the prefix path to the dataset
    :param str data_type: the type of the data, including:
                          - con,
                          - dem,
                          - bird,
                          - impaired,
                          - normal,
                          - wls
                          - add_train_con
                          - add_train_dem
                          - add_test_con
                          - add_test_dem
    :return: the dataframe for the dataset
    :rtype: pandas.DataFrame
    """
    trans_df = pd.DataFrame(columns=["file", "text", "label"])
    for filename in os.listdir(prefix_path):
        if filename.endswith(".txt"):
            if data_type in ("con", "dem"):
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.replace("<s>", "")
                        line = line.replace("</s>", "")
                        line = line.strip()
                        doc_string += line
                        doc_string += ". "
                        doc_string += line
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                    if data_type == "con":
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 0},
                                                   ignore_index=True)
                    else:
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 1},
                                                   ignore_index=True)
            elif data_type == "wls":
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        if line.startswith("SUBJECT:"):
                            line = line[8:]
                            line = line.strip()
                            doc_string += line
                            doc_string += ". "
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": 0},
                                               ignore_index=True)
            elif data_type in ("normal", "impaired"):
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.strip()
                        doc_string += line
                        doc_string += ". "
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                    if data_type == "normal":
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 0},
                                                   ignore_index=True)
                    else:
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 1},
                                                   ignore_index=True)
            elif data_type == "bird":
                # bird
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.strip()
                        doc_string += line
                        doc_string += ". "
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": 1},
                                                ignore_index=True)
            else:
                meta = pd.read_csv('/edata/ADReSS-IS2020-data/meta_data_test.txt', sep=';')
                # address
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.strip()
                        doc_string += line
                        doc_string += ". "
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                if data_type == "add_train_con":
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": 0},
                                                ignore_index=True)
                elif data_type == "add_train_dem":
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": 1},
                                                ignore_index=True)
                else:
                    label = meta[meta['ID   '] == filename[:-4]+' ']['Label '].values[0]
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": label},
                                                ignore_index=True)
    return trans_df


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


def evaluate_model_without_output(test_frame, model, tokenizer):
    """
    evaluate the test dataframe with modeified model,
    save the result to a dataframe and return it
    :param test_frame: the input dataset for evaluation
    :type test_frame: pandas.DataFrame
    :param model: the modified model
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the gpt-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    
    :return: the dataframe with evaluation results
    :rtype: pandas.DataFrame
    """
    res_df = pd.DataFrame(columns=["file", "label", "perplexity"])
    model.eval()
    if USE_GPU:
        model.to(DEVICE)
    for _, row in test_frame.iterrows():
        with torch.no_grad():
            trans = row["text"]
            outputs = model_driver(trans, model, tokenizer)
            # calculate perplexity score
            perp = math.exp(outputs[0].item())
            eval_dict = {"file": row["file"],
                        "label": row["label"],
                        "perplexity": perp}
            res_df = res_df.append(eval_dict, ignore_index=True)
    return res_df


def evaluate_model_with_output(test_frame, model, tokenizer, output_prefix, out_file):
    """
    evaluate the test dataframe with modeified model,
    save the result to a local file

    :param test_frame: the input dataframe for evaluation
    :type test_frame: Pandas.DataFrame
    :param model: the GPT-2 model for evaluation
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :param output_prefix: the folder prefix for the local file
    :type output_prefix: str
    :param out_file: the file name for the local file
    :type out_file: str
    """
    if not os.path.isdir(output_prefix):
        os.mkdir(output_prefix)
    out_file = output_prefix + out_file
    if os.path.exists(out_file):
        sys.stdout.write("file exists, needs to delete it first\n")
        command = "rm " + out_file
        os.system(command)
    test_frame = test_frame.sample(frac=1)
    model.eval()
    if USE_GPU:
        model.to(DEVICE)
    for _, row in test_frame.iterrows():
        with torch.no_grad():
            trans = row["text"]
            outputs = model_driver(trans, model, tokenizer)
            # calculate perplexity score
            perp = math.exp(outputs[0].item())
            eval_dict = {"file": row["file"],
                        "label": row["label"],
                        "perplexity": perp}
            with open(out_file, "a") as out_f:
                json.dump(eval_dict, out_f)
                out_f.write("\n")


def calcualte_accuracy(labels, perp):
    """
    calcualte accuracy given labels and perpelxity scores

    :param labels: the transcript labels
    :type labels: list
    :param perp: the perplexity scores
    :type perp: list
    :return: accuracy and acu
    """
    fpr, tpr, _ = roc_curve(labels, perp)
    fnr = 1 - tpr
    tnr = 1 - fpr
    auc_level = auc(fpr, tpr)
    prevalence = np.count_nonzero(labels)/len(labels)
    eer_point = np.nanargmin(np.absolute((fnr - fpr)))
    tpr_at_eer = tpr[eer_point]
    tnr_at_eer = tnr[eer_point]
    accuracy = tpr_at_eer * prevalence + tnr_at_eer * (1-prevalence)
    return accuracy, auc_level


def calculate_auc_for_diff_model(labels, con_col, dem_col):
    """
    calculate auc for c-d model only
    :param labels: transcript labels, 0 as control, 1 as dementia
    :type labels: Pandas.Series
    :param con_col: control model perpelxity
    :type con_col: Pandas.Series
    :param dem_col: dementia model perplexity
    :type dem_col: Pandas.Series
    :return: the auc for c-d model
    :rtype: float
    """
    labels = labels.values.tolist()
    diff_perp = con_col - dem_col
    diff_perp = diff_perp.values.tolist()
    return calcualte_accuracy(labels, diff_perp)


def calculate_auc_for_ratio_model(labels, con_col, dem_col):
    """
    calculate auc for c/d model only
    :param labels: transcript labels, 0 as control, 1 as dementia
    :type labels: Pandas.Series
    :param con_col: control model perpelxity
    :type con_col: Pandas.Series
    :param dem_col: dementia model perplexity
    :type dem_col: Pandas.Series
    :return: the auc for c-d model
    :rtype: float
    """
    labels = labels.values.tolist()
    ratio_perp = con_col/dem_col
    ratio_perp = ratio_perp.values.tolist()
    return calcualte_accuracy(labels, ratio_perp)


def calculate_auc_for_log_model(labels, con_col, dem_col):
    """
    calculate auc for log(c)-log(d) model only
    :param labels: transcript labels, 0 as control, 1 as dementia
    :type labels: Pandas.Series
    :param con_col: control model perpelxity
    :type con_col: Pandas.Series
    :param dem_col: dementia model perplexity
    :type dem_col: Pandas.Series
    :return: the auc for c-d model
    :rtype: float
    """
    labels = labels.values.tolist()
    log_perp = np.log(con_col) - np.log(dem_col)
    log_perp = log_perp.values.tolist()
    return calcualte_accuracy(labels, log_perp)


def read_json(full_path):
    """
    read json output files and store as pandas dataframe
    :param str full_path: the full path to the json file
    :param str file_type: 
    """
    df_con = pd.read_json(full_path, orient="records", lines=True)
    df_con.columns = ["file", "label", "control"]
    return df_con


def break_attn_heads_by_layer(model, share, layer, style):
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
    :param style: shuffle type, choice between 'zero' or 'shuffle'
    :type style: str
    :return: the modified model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    offset = int(64*(share/100))
    if style == "zero":
        for head in head_offsets:
            for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset] = \
                    model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset].mul(0)
        return model
    else:
        for head in head_offsets:
            for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                np.random.shuffle(model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset] )
        return model


def generate_dem_text(model, tokenizer):
    """
    use the demenia model to generate text and print out the generated text
    use the first 5 sentences for dementia transcript as prompt input,
    generate up tp 120 characters based on the prompt input and modified model

    :param model: the model for dementia simulation
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    """
    dem_padded_text = ".".join(dem_case.split(".")[:5])
    dem_input = tokenizer.encode(dem_padded_text, add_special_tokens=True, return_tensors="pt")
    if USE_GPU:
        dem_input = dem_input.to(DEVICE)
        model = model.to(DEVICE)
    prompt_dem_length = len(tokenizer.decode(dem_input[0], skip_special_tokens=True,
                            clean_up_tokenization_spaces=True))
    dem_output = model.generate(dem_input, max_length=120, do_sample=True, top_k=60, top_p=0.95)
    sys.stdout.write(20*"-")
    sys.stdout.write("\n")
    output = tokenizer.decode(dem_output[0], skip_special_tokens=True)[prompt_dem_length:]
    output = re.sub(r"\s+", " ", output, flags=re.UNICODE)
    sys.stdout.write(output)
    sys.stdout.write("\n")
    sys.stdout.write(20*"-")
    sys.stdout.write("\n")