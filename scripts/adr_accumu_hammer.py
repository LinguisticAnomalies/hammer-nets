'''
script for evaluating cumulative hammer models on ADReSS dataset
'''
import gc
import sys
import os
import logging
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model, accumu_model_driver
from util_fun import calculate_metrics, check_folder, check_file
from util_fun import read_data, generate_texts


def get_adr_data(data_subset):
    """
    get ADReSS training and test set
    return the dataset in dataframe

    :param data_subset: one of the perspective of ADReSS dataset
    :type data_subset: str
    """
    train_con = read_data("/edata/ADReSS-IS2020-data/transcription/train/cc/", "add_train_con")
    train_dem = read_data("/edata/ADReSS-IS2020-data/transcription/train/cd/", "add_train_dem")
    test = read_data("/edata/ADReSS-IS2020-data/transcription/test/", "add_test")
    train = train_con.append(train_dem)
    train = train.sample(frac=1)
    test = test.sample(frac=1)
    if data_subset == "full":
        return train, test
    elif data_subset == "mild":
        train = train[train["mmse"] > 20]
        test = test[test["mmse"] > 20]
        return train, test
    elif data_subset == "slight":
        train = train[train["mmse"] > 24]
        test = test[test["mmse"] > 24]
        return train, test
    else:
        raise ValueError("data subset is not supported")


def print_res(res_dict, model_style, data_subset):
    """
    print out the best auc and associated accuracy results for c-d, c/d model on training set
    return the best configuration for evaluating test set

    :param res_dict: a dictionary contains all evaluation results
    :type res_dict: dict
    :param model_style: diff (c-d) or ratio (c/d) model
    :type model_style: str
    :return: the best configuration
    :rtype: int/list
    :param data_subset: one of the perspective of ADReSS dataset
    :type data_subset: str
    """
    best_auc = max(res_dict[model_style+"_auc"])
    best_index = [index for index, value in enumerate(res_dict[model_style+"_auc"]) \
        if value == best_auc]
    sys.stdout.write("best configuration:\t{}\n".format(best_index))
    for ind in best_index:
        best_accu = res_dict[model_style+"_accu"][ind]
        sys.stdout.write("best {} model index on {} training set:\t{}\n".format(model_style, data_subset,
               ind+1))
        sys.stdout.write("AUC for best {} model on {} training set:\t{}\n".format(model_style, data_subset, 
                   best_auc))
        sys.stdout.write("Accuracy for best {} model on {} training set:\t{}\n".format(model_style, data_subset, 
                     best_accu))
    return best_index


def evaluate_all(df_full, zero_style, share, data_subset="all"):
    """

    :param df_full: entire ADReSS dataset
    :type df_full: pd.DataFrame
    ::param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    :param data_subset: one of the perspective of ADReSS dataset
    :type data_subset: str
    :param data_subset: evaluate all ADReSS dataset, defaults to "all"
    :type data_subset: str, optional
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    con_res = evaluate_model(df_full, model_con, gpt_tokenizer)
    res_dict = {"con_auc": [], "con_accu": [],
                "diff_auc": [], "diff_accu": [],
                "ratio_auc": [], "ratio_accu": []}
    for i in range(1, 13):
        model_dem = accumu_model_driver(model_dem, share, zero_style, i)
        res_dict = calculate_metrics(res_dict, model_dem,
                                     gpt_tokenizer, df_full, con_res)
    del model_dem
    gc.collect()
    sys.stdout.write("====================================\n")
    for model_style in ("diff", "ratio"):
        sys.stdout.write("################################\n")
        sys.stdout.write("share: {}\n".format(share))
        sys.stdout.write("model style:\t{}\n".format(model_style))
        _ = print_res(res_dict, model_style, data_subset)


def evaluate_df(train_df, test_df, zero_style, share, data_subset):
    """
    evaluate function for 1 fold of training and test set

    :param train_df: the training set
    :type train:_df pd.DataFrame
    :param test_df: the test set
    :type test_df: pd.DataFrame
    :param zero_style: the style of zeroing attn heads, supporting 'random','first' and 'shuffle'
    :type zero_style: str
    :param share: the % attention heads to be changed
    :type share: int
    :param data_subset: one of the perspective of ADReSS dataset
    :type data_subset: str
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
    del model_dem
    gc.collect()
    sys.stdout.write("====================================\n")
    for model_style in ("diff", "ratio"):
        sys.stdout.write("################################\n")
        sys.stdout.write("share: {}\n".format(share))
        sys.stdout.write("model style:\t{}\n".format(model_style))
        best_index = print_res(res_dict_train, model_style, data_subset)
        # apply the best configuration on test set
        con_test_res = evaluate_model(test_df, model_con, gpt_tokenizer)
        res_dict_test = {"con_auc": [], "con_accu": [],
                         "diff_auc": [], "diff_accu": [],
                         "ratio_auc": [], "ratio_accu": []}
        # if multiple best indeces exist
        for item in best_index:
            model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
            sys.stdout.write("+++++++++++++++++++++++++++++\n")
            for i in range(1, item+1):
                model_dem = accumu_model_driver(model_dem, share, zero_style, i)
            res_dict_test = calculate_metrics(res_dict_test, model_dem,
                                              gpt_tokenizer, test_df, con_test_res)
            sys.stdout.write("AUC on {} test set on first {} layers:\t{}\n".format(data_subset, item+1,
                       res_dict_test[model_style+"_auc"][0]))
            sys.stdout.write("Accuracy on {} test set on first {} layers:\t{}\n".format(data_subset, item+1,
                                  res_dict_test[model_style+"_accu"][0]))
            # only generate text for full dataset
            if data_subset == "full":
                out_file = "../results/text/adr_layer_{}_share_{}_{}_{}.tsv".format(item, share, zero_style, model_style)
                generate_texts(model_con, model_dem, gpt_tokenizer, out_file)
    sys.stdout.write("====================================\n")
    del model_con, model_dem, gpt_tokenizer
    gc.collect()


def adr_main(zero_style, data_subset):
    """
    main function for hammer model on ADReSS dataset,
    regular train/test split

    :param zero_style: 'first' or 'random' zeroing approach
    :type zero_style: str
    :param data_subset: ADReSS subset selection
    :type data_subset: str
    """
    start_time = datetime.now()
    check_folder("../results/text/")
    check_folder("../results/logs/")
    log_file = "../results/logs/adr_accumu_{}_{}.log".format(zero_style, data_subset)
    check_file(log_file)
    train_df, test_df = get_adr_data(data_subset)
    log_adr = open(log_file, "a")
    sys.stdout = log_adr
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename=log_file)
    for share in (25, 50, 75, 100):
        evaluate_df(train_df, test_df, zero_style, share, data_subset)
    sys.stdout.write("total running time: {}\n".format(datetime.now() - start_time))


def full_conf(zero_style):
    """
    find the best configuration for entire ADReSS dataset

    :param zero_style: 'first' or 'random' zeroing approach
    :type zero_style: str
    """
    start_time = datetime.now()
    check_folder("../results/text/")
    check_folder("../results/logs/")
    log_file = "../results/logs/adr_accumu_{}_all.log".format(zero_style)
    check_file(log_file)
    train_df, test_df = get_adr_data("full")
    df_full = train_df.append(test_df)
    df_full = df_full.sample(frac=1)
    log_adr = open(log_file, "a")
    sys.stdout = log_adr
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename=log_file)
    for share in (25, 50, 75, 100):
        evaluate_all(df_full, zero_style, share, data_subset="all")
    sys.stdout.write("total running time: {}\n".format(datetime.now() - start_time))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    zero_styles = ("first", "random")
    data_subsets = ("full", "mild", "slight")
    for zero_style in zero_styles:
        for data_subset in data_subsets:
            adr_main(zero_style, data_subset)
            full_conf(zero_style)
