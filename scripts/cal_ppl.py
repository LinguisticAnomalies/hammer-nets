'''
calculate the perplexity on best configuration hammer models
    - ADReSS: zeroing the first 50% attention heads on c/d model
    - CCC: zeroing the first 75% attention heads on c/d model
perpleixty scores are calculated with 5-fold cross vadation,
for ADReSS, the 5-fold cross validation perpleixty is compared with trian/test split provided by the dataset
'''
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model, accumu_model_driver
from util_fun import check_folder, check_file
from util_fun import read_data, process_ccc


def merge_df(df_con, df_dem):
    """
    merge two dataframes into one,
    return the merged dataframe

    :param df_con: perplexity dataframe from control model
    :type df_con: pd.DataFrame
    :param df_dem: perplexity dataframe from dementia model
    :type df_dem: pd.DataFrame
    :return: the merged dataframe
    :rtype: pd.DataFrame
    """
    # groupby participant id and calculate perplexity average
    df_con = df_con.groupby(["file", "label"])["perplexity"].mean().reset_index()
    df_dem = df_dem.groupby(["file", "label"])["perplexity"].mean().reset_index()
    full_res = df_con.merge(df_dem, on="file")
    full_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
    full_res = full_res.drop(["discard"], axis=1)
    return full_res


def calculate_ppl(data_type, train_df, test_df, fold):
    """
    evaluate function for CCC training and test set
    with pre-dfined best configuration

    :param data_type: the specific dataset for evaluation
    :type data_type: str
    :param train_df: the training set
    :type train:_df pd.DataFrame
    :param test_df: the test set
    :type test_df: pd.DataFrame
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    :param fold: the n-th fold
    :type fold: int
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    if data_type == "adr":
        best_index = 9
        share = 50
        zero_style = "first"
    elif data_type == "ccc":
        best_index = 7
        share = 75
        zero_style = "first"
    else:
        raise ValueError("data type is not supported.")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = accumu_model_driver(model_dem, share, zero_style, best_index)
    train_con = evaluate_model(train_df, model_con, gpt_tokenizer)
    train_dem = evaluate_model(train_df, model_con, gpt_tokenizer)
    train_full = merge_df(train_con, train_dem)
    test_con = evaluate_model(test_df, model_con, gpt_tokenizer)
    test_dem = evaluate_model(test_df, model_dem, gpt_tokenizer)
    test_full = merge_df(test_con, test_dem)
    train_file = "../results/evals/{}_train_full_{}.tsv".format(data_type, fold)
    test_file = "../results/evals/{}_test_full_{}.tsv".format(data_type, fold)
    check_file(train_file)
    check_file(test_file)
    train_full.to_csv(train_file, sep="\t", index=False)
    test_full.to_csv(test_file, sep="\t", index=False)


def cross_validation(data_type, df_full, n_fold):
    """
    loosely n-fold cross validation for a full transcript dataset

    :param data_type: the specific dataset for evaluation
    :type data_type: str
    :param df_full: the transcript dataset
    :type df_full: pd.DataFrame
    :param n_fold: number of fold for cross validation
    :type n_fold: int
    """
    pid = df_full["file"].unique()
    pid_fold = np.array_split(pid, n_fold)
    # make output as markdown
    for i, array in enumerate(pid_fold):
        sys.stdout.write("---------------------------------\n")
        sys.stdout.write("fold {}...\n".format(i+1))
        sys.stdout.write("training set pid: {}\n".format(array))
        test_df = df_full[df_full["file"].isin(array)]
        train_df = df_full[~df_full["file"].isin(array)]
        calculate_ppl(data_type, train_df, test_df, i)


def ccc_main(n_fold):
    """
    main function for ccc cross validation

    :param n_fold: number of fold for cross validation
    :type n_fold: int
    """
    process_ccc()
    ccc_df = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    cross_validation("ccc", ccc_df, n_fold)


def adress_main(n_fold):
    """
    main function for ADReSS cross validation

    :param n_fold: number of fold for cross validation
    :type n_fold: int

    """
    train_con = read_data("/edata/ADReSS-IS2020-data/transcription/train/cc/", "add_train_con")
    train_dem = read_data("/edata/ADReSS-IS2020-data/transcription/train/cd/", "add_train_dem")
    test = read_data("/edata/ADReSS-IS2020-data/transcription/test/", "add_test")
    train = train_con.append(train_dem)
    df_full = train.append(test)
    df_full = df_full.sample(frac=1)
    cross_validation("adr", df_full, n_fold)


if __name__ == "__main__":
    start_time = datetime.now()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    CV_FOLD = 5
    check_folder("../results/evals")
    sys.stdout.write("####################\n")
    sys.stdout.write("####### CCC ########\n")
    sys.stdout.write("####################\n")
    ccc_main(CV_FOLD)
    sys.stdout.write("####################\n")
    sys.stdout.write("###### ADReSS ######\n")
    sys.stdout.write("####################\n")
    adress_main(CV_FOLD)
    sys.stdout.write("total running time: {}\n".format(datetime.now() - start_time))
