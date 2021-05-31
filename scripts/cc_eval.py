'''
script for evaluating CCC dataset on merged participant id,
5 fold cross validation on CCC dataset to find the "balanced" train/test split,
evaluate transcripts with permuted and cumulative methods,
for cumulative method only
'''


import sys
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import accumu_model_driver, calculate_metrics
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


def str2array(input_str):
    """
    transform the read-in str from dataframe to ndarray

    :param input_str: the input str from dataframe
    :type input_str: str
    """
    tr_list = input_str[1:-1].split(" ")
    tr_list = [item for item in tr_list if item]
    tr_list = [int(item) for item in tr_list]
    return tr_list


def test(df_full, fold_file, zero_style, share, model_con, tokenizer):
    """
    test Serguei's code

    :param df_full: [description]
    :type df_full: [type]
    :param fold_file: [description]
    :type fold_file: [type]
    """
    cv_dict = {"train_con_auc":[], "train_con_accu":[],
                "train_con_cor":[], "train_con_ppl":[],
                "train_dem_auc":[], "train_dem_accu":[],
                "train_dem_cor":[], "train_dem_ppl":[],
                "train_ratio_auc":[], "train_ratio_accu":[],
                "train_ratio_cor":[], "train_ratio_ppl":[],
                "train_norm_auc":[], "train_norm_accu":[],
                "train_norm_cor":[], "train_norm_ppl":[],
                "test_con_auc":[], "test_con_accu":[],
                "test_con_cor":[], "test_con_ppl":[],
                "test_dem_auc":[], "test_dem_accu":[],
                "test_dem_cor":[], "test_dem_ppl":[],
                "test_ratio_auc":[], "test_ratio_accu":[],
                "test_ratio_cor":[], "test_ratio_ppl":[],
                "test_norm_auc":[], "test_norm_accu":[],
                "test_norm_cor":[], "test_norm_ppl":[]}
    for i in range(5):
        cur_dem_train = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 1)]["trainfiles"].values.tolist()[0]
        cur_con_train = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 0)]["trainfiles"].values.tolist()[0]
        cur_dem_test = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 1)]["testfiles"].values.tolist()[0]
        cur_con_test = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 0)]["testfiles"].values.tolist()[0]
        train_fold = str2array(cur_dem_train) + str2array(cur_con_train)
        test_fold = str2array(cur_dem_test) + str2array(cur_con_test)
        # shuffle pid
        np.random.shuffle(train_fold)
        np.random.shuffle(test_fold)
        train_df = df_full[df_full["file"].isin(train_fold)]
        test_df = df_full[df_full["file"].isin(test_fold)]
        # dict for storing results
        train_res = {"con_auc": [], "con_accu": [],
                    "con_cor": [], "con_ppl": [],
                    "dem_auc": [], "dem_accu": [],
                    "dem_cor": [], "dem_ppl": [],
                    "ratio_auc": [], "ratio_accu": [],
                    "ratio_cor": [], "ratio_ppl": [],
                    "norm_auc": [], "norm_accu": [],
                    "norm_cor": [], "norm_ppl": []}
        test_res = {"con_auc": [], "con_accu": [],
                    "con_cor": [], "con_ppl": [],
                    "dem_auc": [], "dem_accu": [],
                    "dem_cor": [], "dem_ppl": [],
                    "ratio_auc": [], "ratio_accu": [],
                    "ratio_cor": [], "ratio_ppl": [],
                    "norm_auc": [], "norm_accu": [],
                    "norm_cor": [], "norm_ppl": []}
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
            if isinstance(v, list): 
                cv_dict["train_"+k].extend(v)
            else:
                cv_dict["train_"+k].append(v)
        for k, v in test_res.items():
            if isinstance(v, list): 
                cv_dict["test_"+k].extend(v)
            else:
                cv_dict["test_"+k].append(v)
    return cv_dict


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
    np.random.shuffle(pid)
    pid_fold = np.array_split(pid, n_fold)
    fold_res = {"train_con_auc":[], "train_con_accu":[],
                "train_con_cor":[], "train_con_ppl":[],
                "train_dem_auc":[], "train_dem_accu":[],
                "train_dem_cor":[], "train_dem_ppl":[],
                "train_ratio_auc":[], "train_ratio_accu":[],
                "train_ratio_cor":[], "train_ratio_ppl":[],
                "train_norm_auc":[], "train_norm_accu":[],
                "train_norm_cor":[], "train_norm_ppl":[],
                "test_con_auc":[], "test_con_accu":[],
                "test_con_cor":[], "test_con_ppl":[],
                "test_dem_auc":[], "test_dem_accu":[],
                "test_dem_cor":[], "test_dem_ppl":[],
                "test_ratio_auc":[], "test_ratio_accu":[],
                "test_ratio_cor":[], "test_ratio_ppl":[],
                "test_norm_auc":[], "test_norm_accu":[],
                "test_norm_cor":[], "test_norm_ppl":[]}
    for i, array in enumerate(pid_fold):
        train_res = {"con_auc": [], "con_accu": [],
                 "con_cor": [], "con_ppl": [],
                 "dem_auc": [], "dem_accu": [],
                 "dem_cor": [], "dem_ppl": [],
                 "ratio_auc": [], "ratio_accu": [],
                 "ratio_cor": [], "ratio_ppl": [],
                 "norm_auc": [], "norm_accu": [],
                 "norm_cor": [], "norm_ppl": []}
        test_res = {"con_auc": [], "con_accu": [],
                    "con_cor": [], "con_ppl": [],
                    "dem_auc": [], "dem_accu": [],
                    "dem_cor": [], "dem_ppl": [],
                    "ratio_auc": [], "ratio_accu": [],
                    "ratio_cor": [], "ratio_ppl": [],
                    "norm_auc": [], "norm_accu": [],
                    "norm_cor": [], "norm_ppl": []}
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
            if isinstance(v, list): 
                fold_res["train_"+k].extend(v)
            else:
                fold_res["train_"+k].append(v)
        for k, v in test_res.items():
            if isinstance(v, list): 
                fold_res["test_"+k].extend(v)
            else:
                fold_res["test_"+k].append(v)
    return fold_res


def print_table(data_name, cv_dict, share, zero_style):
    """
    print the dictionary as markdown table,
    write cv results to local pickle file

    :param data_name: the name of current cv dataset
    :type data_name: str
    :param cv_dict: n-fold cv result
    :type cv_dict: dict
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    """
    # write to pickle file
    out_f = "../results/ppl/cv_accumu_{}_{}_{}.pkl".format(data_name, zero_style, share)
    with open(out_f, "wb") as handle:
        pickle.dump(cv_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    start_time = datetime.now()
    CV_FOLD = 5
    np.random.seed(1234)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    db_folds = pd.read_csv("db_folds.txt")
    db_folds["label"] = np.where(db_folds["label"] == "dem", 1, 0)
    # prob ad 169
    get_db_dataset()
    db = pd.read_csv("data/db.tsv", sep="\t")
    df_full = pd.read_csv("data/adress_full.tsv", sep="\t")
    ccc = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    #sys.stdout.write("| dataset | con AUC (SD)| con ACC (SD) | con r with MMSE (SD)| dem AUC (SD)| dem ACC (SD) | dem r with MMSE (SD)| ratio AUC (SD)| ratio ACC (SD) | ratio r with MMSE (SD)|\n")
    #sys.stdout.write("| - | - | - | - | - | - | - | - | - | - |\n")
    zero_style = "first"
    share = 50
    cv_dict = cross_validation(df_full, zero_style, share, CV_FOLD, model_con, gpt_tokenizer)
    print_table("adr", cv_dict, share, zero_style)
    sys.stdout.write("adr, {}, {} finished\n".format(share, zero_style))
    cv_dict = cross_validation(db, zero_style, share, CV_FOLD, model_con, gpt_tokenizer)
    print_table("db_c", cv_dict, share, zero_style)
    sys.stdout.write("db, Changye's approach, {}, {} finished\n".format(share, zero_style))
    cv_dict = cross_validation(ccc, zero_style, share, CV_FOLD, model_con, gpt_tokenizer)
    print_table("ccc", cv_dict, share, zero_style)
    sys.stdout.write("ccc, {}, {} finished\n".format(share, zero_style))
    sys.stdout.write("total running time: {}\n".format(datetime.now()-start_time))
