'''
script for evaluating CCC dataset on merged participant id,
5 fold cross validation on CCC dataset to find the "balanced" train/test split,
evaluate transcripts with permuted and cumulative methods,
for cumulative method only
'''

import logging
import gc
import sys
import pickle
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model, accumu_model_driver
from util_fun import calculate_metrics, check_folder, check_file
from util_fun import read_data


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


def print_res(res_dict, model_style):
    """
    print out the best auc and associated accuracy results for c-d, c/d model on training set
    return the best configuration for evaluating test set

    :param res_dict: a dictionary contains all evaluation results
    :type res_dict: dict
    :param model_style: diff (c-d) or ratio (c/d) model
    :type model_style: str
    :return: the best configuration
    :rtype: int/list
    """
    best_auc = max(res_dict[model_style+"_auc"])
    best_index = [index for index, value in enumerate(res_dict[model_style+"_auc"]) if value == best_auc]
    sys.stdout.write("best configuration:\t{}\n".format(best_index))
    for ind in best_index:
        best_accu = res_dict[model_style+"_accu"][ind]
        sys.stdout.write("best {} model index on training set:\t{}\n".format(model_style, ind+1))
        sys.stdout.write("AUC for best {} model on training set:\t{}\n".format(model_style, best_auc))
        sys.stdout.write("Accuracy for best {} model on training set:\t{}\n".format(model_style, best_accu))
    return best_index


def evaluate_df(train_df, test_df, zero_style, share):
    """
    evaluate function for CCC training and test set

    :param train_df: the training set
    :type train:_df pd.DataFrame
    :param test_df: the test set
    :type test_df: pd.DataFrame
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    con_res = evaluate_model(train_df, model_con, gpt_tokenizer)
    res_dict_train = {"con_auc": [], "con_accu": [],
                      "diff_auc": [], "diff_accu": [],
                      "ratio_auc": [], "ratio_accu": []}
    for i in range(1, 13):
        model_dem = accumu_model_driver(model_dem, share, zero_style, i)
        res_dict_train = calculate_metrics(res_dict_train, model_dem,
                                           gpt_tokenizer, train_df, con_res)
    sys.stdout.write("====================================\n")
    for model_style in ("diff", "ratio"):
        sys.stdout.write("################################\n")
        sys.stdout.write("model style:\t{}\n".format(model_style))
        best_index = print_res(res_dict_train, model_style)
        # apply the best configuration on test set
        con_test_res = evaluate_model(test_df, model_con, gpt_tokenizer)
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        res_dict_test = {"con_auc": [], "con_accu": [],
                         "diff_auc": [], "diff_accu": [],
                         "ratio_auc": [], "ratio_accu": []}
        # if multiple best indeces exist
        for item in best_index:
            sys.stdout.write("+++++++++++++++++++++++++++++\n")
            for i in range(1, item):
                model_dem = accumu_model_driver(model_dem, share, zero_style, i)
            res_dict_test = calculate_metrics(res_dict_test, model_dem,
                                              gpt_tokenizer, test_df, con_test_res)
            sys.stdout.write("AUC on test set on first {} layers:\t{}\n".format(item+1, res_dict_test[model_style+"_auc"][0]))
            sys.stdout.write("Accuracy on test set on first {} layers:\t{}\n".format(item+1, res_dict_test[model_style+"_accu"][0]))
    sys.stdout.write("====================================\n")
    del model_con, model_dem, gpt_tokenizer
    gc.collect()


def cross_validation(df_full, zero_style, share, n_fold):
    """
    loosely n-fold cross validation for a full transcript dataset

    :param df_full: the transcript dataset
    :type df_full: pd.DataFrame
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    :param n_fold: number of fold for cross validation
    :type n_fold: int
    """
    sys.stdout.write("%:\t{}\n".format(share))
    pid = df_full["file"].unique()
    pid_fold = np.array_split(pid, n_fold)
    # make output as markdown
    for i, array in enumerate(pid_fold):
        sys.stdout.write("---------------------------------\n")
        sys.stdout.write("fold {}...\n".format(i+1))
        sys.stdout.write("training set pid: {}\n".format(array))
        test_df = df_full[df_full["file"].isin(array)]
        train_df = df_full[~df_full["file"].isin(array)]
        evaluate_df(train_df, test_df, zero_style, share)

def ccc_main():
    process_ccc()
    n_fold = 5
    ccc_df = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    check_folder("../results/logs/")
    log_file = "../results/logs/ccc_first_accumu.log"
    check_file(log_file)
    log = open(log_file, "a")
    sys.stdout = log
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename=log_file)
    for share in (25, 50, 75, 100):
        cross_validation(ccc_df, "first", share, n_fold)
    sys.stdout.flush()
    log.close()


def adress_main():
    train_con = read_data("/edata/ADReSS-IS2020-data/transcription/train/cc/", "add_train_con")
    train_dem = read_data(("/edata/ADReSS-IS2020-data/transcription/train/cd/", "add_train_dem"))
    test = read_data(("/edata/ADReSS-IS2020-data/transcription/test/", "add_test"))
    train = train_con.append(train_dem)
    df_full = train.append(test)
    df_full = df_full.sample(frac=1)
    n_fold = 5
    check_folder("../results/logs/")
    log_file = "../results/logs/adr_first_accumu.log"
    check_file(log_file)
    log = open(log_file, "a")
    sys.stdout = log
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename=log_file)
    for share in (25, 50, 75, 100):
        cross_validation(df_full, "first", share, n_fold)
    sys.stdout.flush()
    log.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    start_time = datetime.now()
    sys.stdout.write("####################\n")
    sys.stdout.write("####### CCC ########\n")
    sys.stdout.write("####################\n")
    ccc_main()
    sys.stdout.write("####################\n")
    sys.stdout.write("###### ADReSS ######\n")
    sys.stdout.write("####################\n")
    sys.stdout.write("total running time: {}\n".format(datetime.now() - start_time))
    
