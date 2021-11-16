"""
Utility functions
"""

import os
import gc
import re
import warnings
import pickle
import math
import sys
import pandas as pd
import numpy as np
import torch
from nltk.probability import FreqDist
from sklearn.metrics import roc_curve, auc


warnings.filterwarnings('ignore')
DEVICE = "cuda"
USE_GPU = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def check_folder(folder_path):
    """
    check if folder exists

    :param folder_path: path to the folder
    :type folder_path: str
    """
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def check_file(file_path):
    """
    if the file exists, then delete it

    :param file_path: the full path to the fle
    :type file_path: str
    """
    if os.path.exists(file_path):
        os.system("rm " + file_path)


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


def clean_ccc_text(text):
    """
    basic pre-processing for CCC transcripts
    :param text: the CCC transcript for pre-processing
    :type text: str
    """
    text = text.lower().strip()
    # remove () and [] and following punctuation
    text = re.sub(r"[\(\[].*?[\)\]][?!,.]?", "", text)
    # remove ^^_____ and following punctuation
    text = re.sub(r'\^+_+\s?[?.!,]?', "", text)
    text = re.sub(r'~','-',text)
    text = re.sub(r'-',' ',text)
    text = re.sub(r'[^\x00-\x7F]+','\'',text)
    text = re.sub(r'\<|\>',' ',text)
    text = re.sub(r"\_", "", text)
    # remove unwanted whitespaces between words
    text = re.sub(r'\s+', " ", text)
    text = text.strip()
    return text


def process_ccc():
    """
    read and pre-process ccc dataset:
        - read the ccc dataset pickle file
        - clean the transcript
        - save it to a local file
    :return: ccc cleaned dataset
    :rtype: pd.DataFrame
    """
    with open("/edata/dementia_interviewee_only_0527.pkl", "rb") as f:
        df = pickle.load(f)
    df["label"] = np.where(df["dementia"], 1, 0)
    df = df[["ParticipantID", "Transcript", "label"]]
    # rename columns to keep same track with ADReSS
    df.columns = ["file", "text", "label"]
    df["text"] = df["text"].apply(clean_ccc_text)
    # drop empty rows if any
    df = df[df["text"].str.len() > 0]
    df.to_csv("data/ccc_cleaned.tsv", sep="\t", index=False)


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
        del input_ids, att_mask
        gc.collect()
        return outputs
    else:
        input_ids = encodings["input_ids"]
        att_mask = encodings["attention_mask"]
        outputs = model(input_ids, attention_mask=att_mask, labels=input_ids)
        del input_ids, att_mask
        gc.collect()
        return outputs


def evaluate_model(test_frame, model, tokenizer):
    """
    evaluate the test dataframe with modified model,
    if output is True, then save the results to local file,
    else returns the results as a dataframe

    :param test_frame: the input dataset for evaluation
    :type test_frame: pandas.DataFrame
    :param model: the modified model
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the gpt-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :return: a dataframe
    :rtype: pandas.DataFrame
    """
    model.eval()
    res_df = pd.DataFrame(columns=["file", "label", "perplexity", "mmse"])
    columns = test_frame.columns
    if USE_GPU:
        model.to(DEVICE)
    for _, row in test_frame.iterrows():
        with torch.no_grad():
            trans = row["text"]
            outputs = model_driver(trans, model, tokenizer)
            # calculate perplexity score
            perp = math.exp(outputs[0].item())
            # add MMSE column
            if "mmse" in columns:
                eval_dict = {"file": row["file"],
                             "label": row["label"],
                             "perplexity": perp,
                             "mmse": row["mmse"]}
            else:
                eval_dict = {"file": row["file"],
                             "label": row["label"],
                             "perplexity": perp,
                             "mmse": 0}
            res_df = res_df.append(eval_dict, ignore_index=True)
            del outputs
            gc.collect()
    return res_df


def calculate_accuracy(labels, perp):
    """
    calculate accuracy given labels and perpelxity scores at equal error rate

    :param labels: the transcript labels
    :type labels: list
    :param perp: the perplexity scores
    :type perp: list
    :return: accuracy and auc
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


def calculate_metrics(res_dict, model_dem, tokenizer,
                      input_frame, con_res_df):
    """
    calculate AUC and accuracy for control, dementia, c/d model
    return the evaluation result

    :param res_dict: a dictionary to store all metrics
    :type res_dict: dict
    :param model_dem: the modified model serving as dementia
    :type model_dem: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the gpt-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :param input_frame: the dataframe to be evaluated
    :type input_frame: pd.DataFrame
    :param con_res_df: the groupby control model evaluation result dataframe
    :type con_res_df: pd.DataFrame
    """
    dem_res_df = evaluate_model(input_frame, model_dem, tokenizer)
    dem_res_df = dem_res_df.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
    dem_res_df.rename(columns={"perplexity": "dem_ppl"}, inplace=True)
    full_res = pd.merge(con_res_df, dem_res_df, on=["file", "label", "mmse"])
    mmse = full_res["mmse"].values.tolist()
    # control model AUC, accuracy, cor
    labels = full_res["label"].values.tolist()
    con_ppl = full_res["con_ppl"].values.tolist()
    con_accu, con_auc = calculate_accuracy(labels, con_ppl)
    c_r = np.corrcoef(con_ppl, mmse)[1,0]
    res_dict["con_auc"].append(round(con_auc, 3))
    res_dict["con_accu"].append(round(con_accu, 3))
    res_dict["con_cor"].append(round(c_r, 3))
    res_dict["con_ppl"].append((np.mean(con_ppl)))
    # dementia model AUC, accuracy, cor
    dem_ppl = full_res["dem_ppl"].values.tolist()
    dem_accu, dem_auc = calculate_accuracy(labels, dem_ppl)
    d_r = np.corrcoef(dem_ppl, mmse)[1,0]
    res_dict["dem_auc"].append(round(dem_auc, 3))
    res_dict["dem_accu"].append(round(dem_accu, 3))
    res_dict["dem_cor"].append(round(d_r, 3))
    res_dict["dem_ppl"].append(np.mean(dem_ppl))
    # c/d model AUC, accuracy, cor
    ratio_ppl = full_res["con_ppl"]/full_res["dem_ppl"]
    ratio_ppl = ratio_ppl.values.tolist()
    ratio_accu, ratio_auc = calculate_accuracy(labels, ratio_ppl)
    ratio_r = np.corrcoef(ratio_ppl, mmse)[1,0]
    res_dict["ratio_auc"].append(round(ratio_auc, 3))
    res_dict["ratio_accu"].append(round(ratio_accu, 3))
    res_dict["ratio_cor"].append(round(ratio_r, 3))
    res_dict["ratio_ppl"].append((np.mean(ratio_ppl)))
    # log(c)-log(d) AUC, accuracy, cor
    '''
    log_ppl = np.log(full_res["con_ppl"])-np.log(full_res["dem_ppl"])
    log_ppl = log_ppl.values.tolist()
    log_accu, log_auc = calculate_accuracy(labels, log_ppl)
    log_r = np.corrcoef(log_ppl, mmse)[1,0]
    res_dict["log_auc"].append(round(log_auc, 3))
    res_dict["log_accu"].append(round(log_accu, 3))
    res_dict["log_cor"].append(round(log_r, 3))
    res_dict["log_ppl"].append((np.mean(log_ppl)))
    '''
    # norm ppl diff
    norm_ppl = (np.log(full_res["con_ppl"]) - np.log(full_res["dem_ppl"]))/np.log(full_res["con_ppl"])
    norm_ppl = norm_ppl.values.tolist()
    norm_accu, norm_auc = calculate_accuracy(labels, norm_ppl)
    norm_r = np.corrcoef(norm_ppl, mmse)[1,0]
    res_dict["norm_auc"].append(round(norm_auc, 3))
    res_dict["norm_accu"].append(round(norm_accu, 3))
    res_dict["norm_cor"].append(round(norm_r, 3))
    res_dict["norm_ppl"].append(np.mean(norm_ppl))
    return res_dict


def break_attn_heads_by_layer(zero_type, model, share, layer):
    """
    set certain percentage attention heads to zero at specific layer
    return the modified model
    :param zero_type: the type for zeroing attention heads,
                      'random', 'first' and 'shuffle' are supported
    :type zero_type: str
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
    # zeroing both weights and bias
    head_offsets = [1536, 1536+64, 1536+128, 1536+192, 1536+256,
                    1536+320, 1536+384, 1536+448, 1536+512,
                    1536+576, 1536+640, 1536+704]
    batch = 64
    with torch.no_grad():
        if zero_type == 'random':
            np.random.seed(42)
            torch.manual_seed(42)
            # Serguei's approach to reduce running time
            for head in head_offsets:
                # update to unique random integers
                rnd_index = np.random.choice(range(head, head+64), int(batch*(share/100)), replace=False)
                for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                    model.transformer.h[layer].attn.c_attn.weight[row][rnd_index] = \
                        model.transformer.h[layer].attn.c_attn.weight[row][rnd_index].mul(0.0)
                    model.transformer.h[layer].attn.c_attn.bias[rnd_index] = \
                        model.transformer.h[layer].attn.c_attn.bias[rnd_index].mul(0.0)
            return model
        elif zero_type == 'first':
            offset = int(batch*(share/100))
            for head in head_offsets:
                for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                    model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset] = \
                        model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset].mul(0.0)
                    model.transformer.h[layer].attn.c_attn.bias[head:head+offset] = \
                        model.transformer.h[layer].attn.c_attn.bias[head:head+offset].mul(0.0)
            return model
        elif zero_type == 'shuffle':
            offset = int(64*(share/100))
            for head in head_offsets:
                for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                    np.random.shuffle(model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset])
                    np.random.shuffle(model.transformer.h[layer].attn.c_attn.bias[row][head:head+offset])
            return model
        else:
            raise ValueError("zeroing type is not supported!")


def accumu_model_driver(model, share, zero_style, num_layers):
    """
    the driver function for breaking GPT-2 model
    :param model: the oringal GPT-2 model to be modified
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param share: % of attention heads to be zeroed
    :type share: int
    :param zero_style: the style of zeroing attention heads
    :type zero_style: str
    :param num_layers: numer of layers to be zeroed
    :type num_layers: int
    :return: the modified model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    if num_layers > 13:
        raise ValueError("GPT-2 model only has 12 layers")
    for i in range(0, num_layers):
        model = break_attn_heads_by_layer(zero_style, model, share, i)
    return model


def generate_texts(model_con, model_dem, tokenizer, bird_sents):
    """
    generate additional 20 tokens for each sentence of healthy bird transcript,
    find the highest non-empty beam result for both dementia and control model
    return the output as a dataframe
    :param model_con: the control model
    :type model_con: transformers.modeling_gpt2.GPT2LMHeadModel
    :param model_dem: the dementia model
    :type model_dem: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :param bird_sents: the sentences from bird control transcript
    :type bird_df: list
    """
    torch.manual_seed(42)
    np.random.seed(42)
    # in case of duplicate outputs
    out_df = pd.DataFrame(columns=["sentence", "control", "dementia"])
    # iterate all senteces from healthy bird transcript
    for sent in bird_sents:
        encoded_input = tokenizer.encode(sent, add_special_tokens=True, return_tensors="pt")
        prompt_length = len(tokenizer.decode(encoded_input[0], skip_special_tokens=True,
                            clean_up_tokenization_spaces=True))
        con_output = model_con.generate(encoded_input, top_p=0.9, temperature=1,
                                        max_time=5, num_beams=5, no_repeat_ngram_size=1,
                                        num_return_sequences=5, early_stopping=False,
                                        max_length=prompt_length+20)
        dem_output = model_dem.generate(encoded_input, top_p=0.9, temperature=1,
                                        max_time=5, num_beams=5, no_repeat_ngram_size=1,
                                        num_return_sequences=5, early_stopping=False,
                                        max_length=prompt_length+20)
        # find the non-empty output for both model
        for con_beam, dem_beam in zip(con_output, dem_output):
            # control model output
            con_beam = tokenizer.decode(con_beam, skip_special_tokens=True)[prompt_length:]
            con_beam = re.sub(r"\s+", " ", con_beam, flags=re.UNICODE)
            con_beam = re.sub('"', "", con_beam)
            # dementia model output
            dem_beam = tokenizer.decode(dem_beam, skip_special_tokens=True)[prompt_length:]
            dem_beam = re.sub(r"\s+", " ", dem_beam, flags=re.UNICODE)
            dem_beam = re.sub('"', "", dem_beam)
            # check if outputs are empty
            # both of them are not empty
            if con_beam.strip() and dem_beam.strip():
                out_dict = {"sentence": sent, "control": con_beam, "dementia": dem_beam}
                out_df = out_df.append(out_dict, ignore_index=True)
                break
    del model_con, model_dem
    gc.collect()
    return out_df