'''
script for evaluating CCC dataset on merged participant id,
5 fold cross validation on CCC dataset to find the "balanced" train/test split,
evaluate transcripts with permuted and cumulative methods,
for cumulative method only
'''

import logging
import gc
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model, accumu_model_driver
from util_fun import calculate_metrics, check_folder, check_file
from util_fun import read_data, get_dbca_dataset


def print_res(res_dict, model_style):
    """
    find the best configuration, including best index, AUC and accruacy
    if there are multiple indexes, find the index with highest accuracy

    :param res_dict: a dictionary contains all evaluation results
    :type res_dict: dict
    :param model_style: diff (c-d) or ratio (c/d) model
    :type model_style: str
    :return: the best configuration
    :rtype: int/list
    """
    best_auc = max(res_dict[model_style+"_auc"])
    best_index = [index for index, value in enumerate(res_dict[model_style+"_auc"]) \
        if value == best_auc]
    # if there is multiple best indexes
    if len(best_index) > 1:
        # find the highest accuracy
        accus = [res_dict[model_style+"_accu"][ind] for ind in best_index]
        best_accu = max(accus)
        best_index = [index for index, value in enumerate(accus)\
            if value == best_accu]
    else:
        best_accu = res_dict[model_style+"_accu"][best_index[0]]
    # add 1 for the first n fashion
    best_index = [item+1 for item in best_index]
    return best_index, best_auc, best_accu


def format_res(train_df, test_df, zero_style, model_style, share, fold):
    """
    evaluate the input dataframe and format the results as markdown table,
    return the best AUC and accuracy for train and test fold

    :param train_df: the input train fold dataframe
    :type train_df: pd.DataFrame
    :param test_df: the input test fold dataframe
    :type test_df: pd.DataFrame
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param model_style: c/d ('ratio') or c-d ('diff) model for evaluation
    :type model_style: str
    :param share: the % attention heads to be changed
    :type share: int
    :param fold: index of current fold
    :type fold: int
    rtype: float, float, float, float
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    # find the best configuration on the train fold
    con_res = evaluate_model(train_df, model_con, gpt_tokenizer)
    res_dict_train = {"con_auc": [], "con_accu": [],
                      "diff_auc": [], "diff_accu": [],
                      "ratio_auc": [], "ratio_accu": []}
    res_dict_test = {"con_auc": [], "con_accu": [],
                     "diff_auc": [], "diff_accu": [],
                     "ratio_auc": [], "ratio_accu": []}
    for i in range(1, 13):
        model_dem = accumu_model_driver(model_dem, share, zero_style, i)
        res_dict_train = calculate_metrics(res_dict_train, model_dem,
                                           gpt_tokenizer, train_df, con_res)
    best_index, best_train_auc, best_train_accu = print_res(res_dict_train, model_style)
    # evaluate the best configuration on test fold
    con_test_res = evaluate_model(test_df, model_con, gpt_tokenizer)
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = accumu_model_driver(model_dem, share, zero_style, best_index[0])
    res_dict_test = calculate_metrics(res_dict_test, model_dem,
                                      gpt_tokenizer, test_df, con_test_res)
    _, best_test_auc, best_test_accu = print_res(res_dict_test, model_style)
    sys.stdout.write("| {} | {} | {} ({}) | {} ({})|\n".format(fold+1, best_index, best_train_auc, best_train_accu, best_test_auc, best_test_accu))
    del gpt_tokenizer, model_con, model_dem
    gc.collect()
    return best_train_auc, best_train_accu, best_test_auc, best_test_accu  


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
    sys.stdout.write("zero style:\t{}\n".format(zero_style))
    pid = df_full["file"].unique()
    pid_fold = np.array_split(pid, n_fold)
    eval_res = {"train_auc_diff":[], "train_auc_ratio":[],
                "train_accu_diff":[], "train_accu_ratio":[],
                "test_auc_diff":[], "test_auc_ratio":[],
                "test_accu_diff": [], "test_accu_ratio":[]}
    # make output as markdown
    for model_style in ("diff", "ratio"):
        sys.stdout.write("model style:\t{}\n".format(model_style))
        sys.stdout.write("| fold | best index | train AUC (Accuracy) | test AUC (Accuracy) |\n")
        sys.stdout.write("| - | - | - | - |\n")
        for i, array in enumerate(pid_fold):
            test_df = df_full[df_full["file"].isin(array)]
            train_df = df_full[~df_full["file"].isin(array)]
            train_auc, train_accu, test_auc, test_accu = format_res(train_df, test_df, zero_style, model_style, share, i)
            eval_res["train_auc_"+model_style].append(train_auc)
            eval_res["train_accu_"+model_style].append(train_accu)
            eval_res["test_auc_"+model_style].append(test_auc)
            eval_res["test_accu_"+model_style].append(test_accu)
        # print out averaged evaluation results
        train_avg_auc = round(np.mean(eval_res["train_auc_"+model_style]), 2)
        train_avg_accu = round(np.mean(eval_res["train_accu_"+model_style]), 2)
        test_avg_auc = round(np.mean(eval_res["test_auc_"+model_style]), 2)
        test_avg_accu = round(np.mean(eval_res["test_accu_"+model_style]), 2)
        sys.stdout.write("| average | NA | {} ({})| {} ({}) |\n".format(train_avg_auc, train_avg_accu, test_avg_auc, test_avg_accu))

def ccc_main(n_fold, zero_style):
    """
    main function for ccc cross validation

    :param n_fold: number of fold for cross validation
    :type n_fold: int
    :param zero_style: the style of zeroing attention heads
    :type zero_style: str
    """
    ccc_df = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    sys.stdout.write("####################\n")
    sys.stdout.write("####### CCC ########\n")
    sys.stdout.write("####################\n")
    for share in (25, 50, 75, 100):
        cross_validation(ccc_df, zero_style, share, n_fold)


def dbca_main(n_fold, zero_style):
    """
    main function for DBCA cross validation

    :param n_fold: number of fold for cross validation
    :type n_fold: int
    :param zero_style: the style of zeroing attention heads
    :type zero_style: str
    """
    df_full = get_dbca_dataset()
    sys.stdout.write("####################\n")
    sys.stdout.write("####### DBCA #######\n")
    sys.stdout.write("####################\n")
    for share in (25, 50, 75, 100):
        cross_validation(df_full, zero_style, share, n_fold)


def adr_main(n_fold, zero_style):
    """
    main function for ADReSS cross validation

    :param n_fold: number of fold for cross validation
    :type n_fold: int
    :param zero_style: the style of zeroing attention heads
    :type zero_style: str
    """
    if os.path.exists("data/adress_full.tsv"):
        df_full = pd.read_csv("data/adress_full.tsv", sep="\t")
    else:
        train_con = read_data("/edata/ADReSS-IS2020-data/transcription/train/cc/", "add_train_con")
        train_dem = read_data("/edata/ADReSS-IS2020-data/transcription/train/cd/", "add_train_dem")
        test = read_data("/edata/ADReSS-IS2020-data/transcription/test/", "add_test")
        train = train_con.append(train_dem)
        df_full = train.append(test)
        df_full = df_full.sample(frac=1)
        df_full.to_csv("data/adress_full.tsv", sep="\t", index=False)
    check_folder("../results/logs/")
    # three subsets
    sys.stdout.write("####################\n")
    sys.stdout.write("### ADReSS full ####\n")
    sys.stdout.write("####################\n")
    for share in (25, 50, 75, 100):
        cross_validation(df_full, zero_style, share, n_fold)
    mild = df_full[df_full["mmse"] > 20]
    sys.stdout.write("####################\n")
    sys.stdout.write("### ADReSS mild ####\n")
    sys.stdout.write("####################\n")
    for share in (25, 50, 75, 100):
        cross_validation(mild, zero_style, share, n_fold)
    # slight
    slight = df_full[df_full["mmse"] > 24]
    sys.stdout.write("####################\n")
    sys.stdout.write("### ADReSS slight ##\n")
    sys.stdout.write("####################\n")
    for share in (25, 50, 75, 100):
        cross_validation(slight, zero_style, share, n_fold)


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    check_folder("../results/logs/")
    log_file = "../results/logs/accumu_all.log"
    check_file(log_file)
    log_adr = open(log_file, "a")
    sys.stdout = log_adr
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename=log_file)
    CV_FOLD = 5
    zero_styles = ("first", "random")
    for zero_style in zero_styles:
        ccc_main(CV_FOLD, zero_style)
        dbca_main(CV_FOLD, zero_style)
        adr_main(CV_FOLD, zero_style)
