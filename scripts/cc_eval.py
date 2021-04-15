'''
script for evaluating CCC dataset on merged participant id,
5 fold cross validation on CCC dataset to find the "balanced" train/test split,
evaluate transcripts with permuted and cumulative methods,
for cumulative method only
'''

import logging
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import accumu_model_driver
from util_fun import calculate_metrics, check_folder, check_file
from util_fun import evaluate_model, get_db_dataset


def print_res(res_dict):
    """
    find the best configuration, including best index, AUC and accruacy
    if there are multiple indexes, find the index with highest accuracy

    :param res_dict: a dictionary contains all evaluation results
    :type res_dict: dict
    :return: the best configuration
    :rtype: int/list
    """
    best_auc = max(res_dict["ratio_auc"])
    best_index = [index for index, value in enumerate(res_dict["ratio_auc"]) \
        if value == best_auc]
    # if there is multiple best indexes
    if len(best_index) > 1:
        # find the highest accuracy
        accus = [res_dict["ratio_accu"][ind] for ind in best_index]
        best_accu = max(accus)
        best_index = [index for index, value in enumerate(accus)\
            if value == best_accu]
    else:
        best_accu = res_dict["ratio_accu"][best_index[0]]
    return best_index


def cross_validation(df_full, zero_style, share,
                     n_fold, model_con, tokenizer):
    """
    loosely n-fold cross validation for a full transcript dataset,
    return fold evaluation result dictionary

    :param df_full: the transcript dataset
    :type df_full: pd.DataFrame
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    :param n_fold: number of fold for cross validation
    :type n_fold: int
    :param model_con: the control model
    :type model_con: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :return: the n-fold cv result dictionary
    :rtype: dict
    """
    pid = df_full["file"].unique()
    pid_fold = np.array_split(pid, n_fold)
    fold_res = {"train_con_auc":[], "train_con_accu":[],
                "train_con_cor":[], "train_con_ppl":[],
                "train_dem_auc":[], "train_dem_accu":[],
                "train_dem_cor":[], "train_dem_ppl":[],
                "train_ratio_auc":[], "train_ratio_accu":[],
                "train_ratio_cor":[], "train_ratio_ppl":[],
                "test_con_auc":[], "test_con_accu":[],
                "test_con_cor":[], "test_con_ppl":[],
                "test_dem_auc":[], "test_dem_accu":[],
                "test_dem_cor":[], "test_dem_ppl":[],
                "test_ratio_auc":[], "test_ratio_accu":[],
                "test_ratio_cor":[], "test_ratio_ppl":[]}
    for i, array in enumerate(pid_fold):
        train_res = {"con_auc": [], "con_accu": [],
                     "con_cor": [], "con_ppl": [],
                     "dem_auc": [], "dem_accu": [],
                     "dem_cor": [], "dem_ppl": [],
                     "ratio_auc": [], "ratio_accu": [],
                     "ratio_cor": [], "ratio_ppl": []}
        test_res = {"con_auc": [], "con_accu": [],
                    "con_cor": [], "con_ppl": [],
                    "dem_auc": [], "dem_accu": [],
                    "dem_cor": [], "dem_ppl": [],
                    "ratio_auc": [], "ratio_accu": [],
                    "ratio_cor": [], "ratio_ppl": []}
        test_df = df_full[df_full["file"].isin(array)]
        train_df = df_full[~df_full["file"].isin(array)]
        # control model evaluation results
        con_res_df = evaluate_model(train_df, model_con, tokenizer)
        con_res_df = con_res_df.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
        con_res_df.rename(columns={"perplexity": "con_ppl"}, inplace=True)
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        # find the best configuration
        for i in range(1, 13):
            model_dem = accumu_model_driver(model_dem, share, zero_style, i)
            train_res = calculate_metrics(train_res,
                                          model_dem, tokenizer,train_df, con_res_df)
        best_train_index = print_res(train_res)
        best_train_index = best_train_index[0]
        sys.stdout.write("best index:\t{}\n".format(best_train_index+1))
        best_train_dict = {}
        # narrow down to the best result
        for k, v in train_res.items():
            if isinstance(v, list):
                best_train_dict[k] = v[best_train_index]
            else:
                best_train_dict[k] = v
        # evaluate the dementia model
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        model_dem = accumu_model_driver(model_dem, share, zero_style, best_train_index+1)
        con_res_df_test = evaluate_model(test_df, model_con, tokenizer)
        con_res_df_test = con_res_df_test.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
        con_res_df_test.rename(columns={"perplexity": "con_ppl"}, inplace=True)
        test_res = calculate_metrics(test_res, model_dem,
                                     tokenizer, test_df, con_res_df_test)
        # add to fold dictionary
        for k, v in best_train_dict.items():
            fold_res["train_"+k].append(v)
        for k, v in test_res.items():
            if isinstance(v, list):
                fold_res[k] = v[0]
            else:
                fold_res[k] = v
    return fold_res


def print_table(data_name, cv_dict):
    """
    print the dictionary as markdown table

    :param data_name: the name of current cv dataset
    :type data_name: str
    :param cv_dict: n-fold cv result
    :type cv_dict: dict
    """
    #sys.stdout.write("| dataset | mmse (control/dementia)| con AUC (SD)| con ACC (SD) | con r with MMSE (SD)| dem AUC (SD)| dem ACC (SD) | dem r with MMSE (SD)| ratio AUC (SD)| ratio ACC (SD) | ratio r with MMSE (SD)|\n")
    #sys.stdout.write("| - | - | - | - | - | - | - | - | - | - | - |\n")
    sys.stdout.write("| {} | 0/0 | {} ({})| {} ({}) | {} ({})| {} ({})| {} ({}) | {} ({})| {} ({})| {} ({}) | {} ({})|\n".format(
        data_name,
        np.mean(cv_dict["con_auc"]), np.std(cv_dict["con_auc"]),
        np.mean(cv_dict["con_accu"]), np.std(cv_dict["con_accu"]),
        np.mean(cv_dict["con_cor"]), np.std(cv_dict["con_cor"]),
        np.mean(cv_dict["dem_auc"]), np.std(cv_dict["dem_auc"]),
        np.mean(cv_dict["dem_accu"]), np.std(cv_dict["dem_accu"]),
        np.mean(cv_dict["dem_cor"]), np.std(cv_dict["dem_cor"]),
        np.mean(cv_dict["ratio_auc"]), np.std(cv_dict["ratio_auc"]),
        np.mean(cv_dict["ratio_accu"]), np.std(cv_dict["ratio_accu"]),
        np.mean(cv_dict["ratio_cor"]), np.std(cv_dict["ratio_cor"])
    ))


if __name__ == "__main__":
    start_time = datetime.now()
    CV_FOLD = 5
    np.random.seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    check_folder("../results/logs/")
    log_file = "../results/logs/accumu_full_comb.log"
    check_file(log_file)
    log_adr = open(log_file, "a")
    sys.stdout = log_adr
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename=log_file)
    df_full = pd.read_csv("data/adress_full.tsv", sep="\t")
    db = get_db_dataset()
    ccc = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    for zero_style in ("first", "random"):
        for share in (25, 50, 75, 100):
            sys.stdout.write("==================================\n")
            sys.stdout.write("zero style:\t{}\n".format(zero_style))
            sys.stdout.write("share:\t{}\n".format(share))
            cv_dict = cross_validation(df_full, zero_style, share, CV_FOLD, model_con, gpt_tokenizer)
            print_table("ADReSS", cv_dict)
            cv_dict = cross_validation(db, zero_style, share, CV_FOLD, model_con, gpt_tokenizer)
            print_table("DemBank", cv_dict)
            cv_dict = cross_validation(ccc, zero_style, share, CV_FOLD, model_con, gpt_tokenizer)
            sys.stdout.write("| dataset | mmse (control/dementia)| con AUC (SD)| con ACC (SD) | con r with MMSE (SD)| dem AUC (SD)| dem ACC (SD) | dem r with MMSE (SD)| ratio AUC (SD)| ratio ACC (SD) | ratio r with MMSE (SD)|\n")
            sys.stdout.write("| - | - | - | - | - | - | - | - | - | - | - |\n")
            print_table("CCC", cv_dict)
            sys.stdout.write("==================================\n")
            sys.stdout.write("\n")
    sys.stdout.write("total running time: {}\n".format(datetime.now()-start_time))
