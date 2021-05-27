'''
Find the best configuration on different dataset
For cumulative method, c/d model only
'''
import sys
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model, accumu_model_driver
from util_fun import calculate_metrics


def print_res(res_dict, model_type):
    """
    find the best configuration, including best index, AUC and accruacy
    if there are multiple indexes, find the index with highest accuracy

    :param res_dict: a dictionary contains all evaluation results
    :type res_dict: dict
    :param model_type: the model type, including 'ratio' and 'norm'
    :type model_type: str
    :return: the best configuration
    :rtype: int/list
    """
    best_auc = max(res_dict[model_type+"_auc"])
    best_index = [index for index, value in enumerate(res_dict["ratio_auc"]) \
        if value == best_auc]
    # if there is multiple best indexes
    if len(best_index) > 1:
        # find the highest accuracy
        accus = [res_dict[model_type+"_accu"][ind] for ind in best_index]
        best_accu = max(accus)
        best_index = [index for index, value in enumerate(accus)\
            if value == best_accu]
    else:
        best_accu = res_dict[model_type+"_accu"][best_index[0]]
    return best_index


def find_best_train(data_name, model_con, tokenizer,
                    share, zero_style):
    """
    find the best configuration, return the best metrics

    :param data_name: the name of dataset
    :type data_name: str
    :param model_con: the control model
    :type model_con: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    """
    start_time = datetime.now()
    train_res = {"con_auc": [], "con_accu": [],
                 "con_cor": [], "con_ppl": [],
                 "dem_auc": [], "dem_accu": [],
                 "dem_cor": [], "dem_ppl": [],
                 "ratio_auc": [], "ratio_accu": [],
                 "ratio_cor": [], "ratio_ppl": [],
                 "norm_auc":[], "norm_accu":[],
                 "norm_cor":[], "norm_ppl":[]}
    if data_name == "adr":
        train_df = pd.read_csv("data/adress_full.tsv", sep="\t")
    elif data_name == "adr_train":
        train_df = pd.read_csv("data/adress_train_full.tsv", sep="\t")
    elif data_name == "adr_test":
        train_df = pd.read_csv("data/adress_test_full.tsv", sep="\t")
    elif data_name == "ccc":
        train_df = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    elif data_name == "db":
        train_df = pd.read_csv("data/db.tsv", sep="\t")
    elif data_name == "db_full":
        train_df = pd.read_csv("data/db_full.tsv", sep="\t")
    else:
        raise ValueError("data name is not supported.")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    con_res_df = evaluate_model(train_df, model_con, tokenizer)
    con_res_df = con_res_df.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
    con_res_df.rename(columns={"perplexity": "con_ppl"}, inplace=True)
    for i in range(1, 13):
        model_dem = accumu_model_driver(model_dem, share, zero_style, i)
        train_res = calculate_metrics(train_res, model_dem,
                                      tokenizer,train_df, con_res_df)
    # write to pickle file
    out_f = "../results/ppl/accumu_{}_{}_{}.pkl".format(data_name, zero_style, share)
    with open(out_f, "wb") as handle:
        pickle.dump(train_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.write("total running time: {}\n".format(datetime.now() - start_time))
        

def cross_validation(base_df, test_df, model_con,
                     model_dem, tokenizer):
    """
    loosely n-fold cross validation for a full transcript dataset,
    return fold evaluation result dictionary

    :param base_df: the training set for finding the best conf
    :type base_df: pd.DataFrame
    :param test_df: the test set for evaluation
    :type base_df: pd.DataFrame
    :param model_con: the control model
    :type model_con: transformers.modeling_gpt2.GPT2LMHeadModel
    :param model_dem: the dementia model
    :type model_dem: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :return: the n-fold cv result dictionary
    :rtype: dict
    """
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
    con_res_df = evaluate_model(base_df, model_con, tokenizer)
    con_res_df = con_res_df.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
    con_res_df.rename(columns={"perplexity": "con_ppl"}, inplace=True)
    con_res_df_test = evaluate_model(test_df, model_con, tokenizer)
    con_res_df_test = con_res_df_test.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
    con_res_df_test.rename(columns={"perplexity": "con_ppl"}, inplace=True)
    train_res = calculate_metrics(train_res, model_dem,
                                  tokenizer, base_df, con_res_df)
    test_res = calculate_metrics(test_res, model_dem,
                                 tokenizer, test_df, con_res_df_test)
    return train_res, test_res


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


def main_driver(model_con, tokenizer):
    """
    the driver function for cross validation,
    apply ADReSS best configuration on CCC and DB dataset
    """
    db_full = pd.read_csv("data/db_full.tsv", sep="\t")
    ccc = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    adr_full = pd.read_csv("data/adress_full.tsv", sep="\t")
    # best configuration on full ADReSS dataset
    zero_style = "first"
    share = 50
    layers = 9
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = accumu_model_driver(model_dem, share, zero_style, layers)
    sys.stdout.write("| dataset | mmse (control/dementia)| con AUC (SD)| con ACC (SD) | con r with MMSE (SD)| dem AUC (SD)| dem ACC (SD) | dem r with MMSE (SD)| ratio AUC (SD)| ratio ACC (SD) | ratio r with MMSE (SD)|\n")
    sys.stdout.write("| - | - | - | - | - | - | - | - | - | - | - |\n")
    train_res, test_res = cross_validation(ccc, db_full, model_con,
                                           model_dem, tokenizer)
    print_table("ccc", train_res)
    print_table("db_full", test_res)
    sys.stdout.write("\n")
    # best configuration on db full
    sys.stdout.write("| dataset | mmse (control/dementia)| con AUC (SD)| con ACC (SD) | con r with MMSE (SD)| dem AUC (SD)| dem ACC (SD) | dem r with MMSE (SD)| ratio AUC (SD)| ratio ACC (SD) | ratio r with MMSE (SD)|\n")
    sys.stdout.write("| - | - | - | - | - | - | - | - | - | - | - |\n")
    share = 50
    layers = 9
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = accumu_model_driver(model_dem, share, zero_style, layers)
    train_res, test_res = cross_validation(adr_full, ccc, model_con,
                                           model_dem, tokenizer)
    print_table("adr", train_res)
    print_table("ccc", test_res)
    # best configuration on ccc
    sys.stdout.write("\n")
    sys.stdout.write("| dataset | mmse (control/dementia)| con AUC (SD)| con ACC (SD) | con r with MMSE (SD)| dem AUC (SD)| dem ACC (SD) | dem r with MMSE (SD)| ratio AUC (SD)| ratio ACC (SD) | ratio r with MMSE (SD)|\n")
    sys.stdout.write("| - | - | - | - | - | - | - | - | - | - | - |\n")
    # best configuration on db
    share = 100
    layers = 9
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = accumu_model_driver(model_dem, share, zero_style, layers)
    train_res, test_res = cross_validation(adr_full, db_full, model_con,
                                           model_dem, tokenizer)
    print_table("adr", train_res)
    print_table("db_full", test_res)


if __name__ == "__main__":
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    zero_style = "first"
    share = 50
    find_best_train("adr", model_con, tokenizer,
                    share, zero_style)
    find_best_train("adr_train", model_con, tokenizer,
                    share, zero_style)
    find_best_train("adr_test", model_con, tokenizer,
                    share, zero_style)
    find_best_train("ccc", model_con, tokenizer,
                    share, zero_style)
    find_best_train("db", model_con, tokenizer,
                    share, zero_style)
    #main_driver(model_con, tokenizer)