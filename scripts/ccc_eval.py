'''
script for evaluating CCC dataset on merged participant id,
5 fold cross validation on CCC dataset to find the "balanced" train/test split,
evaluate transcripts with permuted and cumulative methods
'''

import logging
import sys
import pickle
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model, break_attn_heads_by_layer
from util_fun import calculate_metrics, check_folder, check_file


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
    with open("/edata/dementia_cleaned_withId.pkl", "rb") as f:
        df = pickle.load(f)
    df["label"] = np.where(df["dementia"], 1, 0)
    df = df[["ParticipantID", "Transcript", "label"]]
    # rename columns to keep same track with ADReSS
    df.columns = ["file", "text", "label"]
    df["text"] = df["text"].apply(clean_ccc_text)
    # drop empty rows if any
    df = df[df["text"].str.len() > 0]
    return df
    # df.to_csv("data/ccc_cleaned.tsv", sep="\t", index=False)


def split_dataset(ccc_df):
    """
    split the cleaned dataset into training and test set
    return the training and test set

    :param ccc_df: cleaned ccc transcript dataset
    :type ccc_df: pd.DataFrame
    :return: training and test set
    :rtype: pd.DataFrame, pd.DataFrame
    """
    pid = ccc_df["file"].unique()
    # randomly select 30% pid as test set
    pid_test = np.random.choice(pid, size=int(len(pid)*0.3), replace=False)
    test_df = ccc_df.loc[ccc_df["file"].isin(pid_test)]
    train_df = ccc_df.loc[~ccc_df["file"].isin(pid_test)]
    return train_df, test_df


def print_res(res_dict, hammer_style):
    """
    print out the best auc and accu results for c-d, c/d model

    :param res_dict: a dictionary contains all evaluation results
    :type res_dict: dict
    :param hammer_style: the style of changing attention heads, including oneime, accumu and combo
    :type hammer_style: str
    """
    # c-d model
    max_diff_auc = max(res_dict["diff_auc"])
    max_diff_auc_index = [index for index, value in enumerate(res_dict["diff_auc"]) if value == max_diff_auc]
    max_diff_accu = max(res_dict["diff_accu"])
    max_diff_accu_index = [index for index, value in enumerate(res_dict["diff_accu"]) if value == max_diff_accu]
    # c/d model
    max_ratio_auc = max(res_dict["ratio_auc"])
    max_ratio_auc_index = [index for index, value in enumerate(res_dict["ratio_auc"]) if value == max_ratio_auc]
    max_ratio_accu = max(res_dict["ratio_accu"])
    max_ratio_accu_index = [index for index, value in enumerate(res_dict["ratio_accu"]) if value == max_ratio_accu]
    if hammer_style == "accumu":
        sys.stdout.write("best c-d auc index: {}\n".format([x+1 for x in max_diff_auc_index]))
        sys.stdout.write("best c-d auc value: {}\n".format(max_diff_auc))
        sys.stdout.write("best c-d accu index: {}\n".format([x+1 for x in max_diff_accu_index]))
        sys.stdout.write("best c-d accu value: {}\n".format(max_diff_accu))
        sys.stdout.write("best c/d auc index: {}\n".format([x+1 for x in max_ratio_auc_index]))
        sys.stdout.write("best c/d auc value: {}\n".format(max_ratio_auc))
        sys.stdout.write("best c/d accu index: {}\n".format([x+1 for x in max_ratio_accu_index]))
        sys.stdout.write("best c/d accu value: {}\n".format(max_ratio_accu))
    else:
        sys.stdout.write("best c-d auc index: {}\n".format(max_diff_auc_index))
        sys.stdout.write("best c-d auc value: {}\n".format(max_diff_auc))
        sys.stdout.write("best c-d accu index: {}\n".format(max_diff_accu_index))
        sys.stdout.write("best c-d accu value: {}\n".format(max_diff_accu))
        sys.stdout.write("best c/d auc index: {}\n".format(max_ratio_auc_index))
        sys.stdout.write("best c/d auc value: {}\n".format(max_ratio_auc))
        sys.stdout.write("best c/d accu index: {}\n".format(max_ratio_accu_index))
        sys.stdout.write("best c/d accu value: {}\n".format(max_ratio_accu))


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


def evaluate_ccc(ccc_df, hammer_style, zero_style, share):
    """
    evaluate the full ccc dataset

    :param ccc_df: the ccc dataset
    :type ccc_df: pd.DataFrame
    :param hammer_style: the style of changing attention heads, including oneime, accumu and combo
    :type hammer_style: str
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    con_res = evaluate_model(ccc_df, model_con, gpt_tokenizer)
    res_dict = {"con_auc": [], "con_accu": [],
                "diff_auc": [], "diff_accu": [],
                "ratio_auc": [], "ratio_accu": []}
    if hammer_style == "accumu":
        for i in range(1, 13):
            model_dem = accumu_model_driver(model_con, share, zero_style, i)
            res_dict = calculate_metrics(res_dict, model_dem,
                                         gpt_tokenizer, ccc_df, con_res)
    elif hammer_style == "onetime":
        for i in range(0, 12):
            model_con = GPT2LMHeadModel.from_pretrained("gpt2")
            model_dem = break_attn_heads_by_layer(zero_style, model_con, share, i)
            res_dict = calculate_metrics(res_dict, model_dem,
                                         gpt_tokenizer, ccc_df, con_res)
    elif hammer_style == "comb":
        layers = [0, 1, 2, 3, 4, 8, 10]
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        for layer in layers:
            model_dem = break_attn_heads_by_layer(zero_style, model_dem, share, layer)
        res_dict = calculate_metrics(res_dict, model_dem,
                                     gpt_tokenizer, ccc_df, con_res)
    else:
        raise ValueError("Wrong hammer style")
    sys.stdout.write("========================\n")
    sys.stdout.write("hammer style: {}\n".format(hammer_style))
    sys.stdout.write("zero style: {}\n".format(zero_style))
    sys.stdout.write("share: {}\n".format(share))
    for k, v in res_dict.items():
        sys.stdout.write("{}:\t{}\n".format(k, v))
        sys.stdout.flush()
    print_res(res_dict, hammer_style)
    sys.stdout.write("========================\n")


def cross_validation(ccc_df, hammer_style, zero_style, share, n_fold):
    """
    loosely n-fold cross validation for CCC dataset

    :param ccc_df: the ccc dataset
    :type ccc_df: pd.DataFrame
    :param hammer_style: the style of changing attention heads, including oneime, accumu and combo
    :type hammer_style: str
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    :param n_fold: [description]
    :type n_fold: [type]
    """
    pid = ccc_df["file"].unique()
    pid_fold = np.array_split(pid, n_fold)
    for i, array in enumerate(pid_fold):
        sys.stdout.write("---------------------------------\n")
        sys.stdout.write("fold {}...\n".format(i+1))
        train_df = ccc_df[ccc_df["file"].isin(array)]
        test_df = ccc_df[~ccc_df["file"].isin(array)]
        sys.stdout.write("evaluating training set...\n")
        evaluate_ccc(train_df, hammer_style, zero_style, share)
        sys.stdout.write("evaluating testing set...\n")
        evaluate_ccc(test_df, hammer_style, zero_style, share)


if __name__ == "__main__":
    start_time = datetime.now()
    process_ccc()
    n_fold = 5
    ccc_df = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    check_folder("../results/logs/")
    log_file = "../results/logs/ccc_eva.log"
    check_file(log_file)
    log = open(log_file, "a")
    sys.stdout = log
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename=log_file)
    for hammer_style in ("accumu", "onetime", "comb"):
        for share in (25, 50, 75, 100):
            for zero_style in ("first", "random"):
                sys.stdout.write("---------------------------------\n")
                sys.stdout.write("evaluating full ccc dataset\n")
                evaluate_ccc(ccc_df, hammer_style, zero_style, share)
                cross_validation(ccc_df, hammer_style, zero_style, share, n_fold)
    sys.stdout.write("total running time: {}\n".format(datetime.now() - start_time))
    sys.stdout.flush()
    log.close()
