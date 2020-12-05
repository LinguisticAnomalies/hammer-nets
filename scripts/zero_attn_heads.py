"""
Zeroing attention heads in each layer, in terms of ont-time or accumulately.
"""


import sys
import argparse
import warnings
import logging
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model_without_output, generate_dem_text, read_json
from util_fun import calculate_auc_for_diff_model, calculate_auc_for_ratio_model
from util_fun import calculate_auc_for_log_model
from util_fun import calcualte_accuracy, break_attn_heads_by_layer
from util_fun import check_file


gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
train_frame = pd.read_csv("address_train.csv")
test_frame = pd.read_csv("address_test.csv")
con_train_res = read_json("../results/cache-original/con_train.json")
con_test_res = read_json("../results/cache-original/con_test.json")


def parse_args():
    """
    add zeroing head argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str,
                        help="the style of changing attention heads")
    parser.add_argument("--share", type=int,
                        help="the % attention heads to be changed")
    parser.add_argument('--text', type=str2bool,
                        help="boolean indicator for text generation with dementia model")
    return parser.parse_args()


def str2bool(v):
    """
    convert user input into boolean value

    :param v: user input of true or false
    :type v: str
    :raises argparse.ArgumentTypeError: Boolean value expected.
    :return: a boolean value
    :rtype: bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calculate_aucs(eva_method, full_res):
    """
    calcualte different aucs with given folder prefix and evaluation method

    :param eva_method: the evaluation method
    :type eva_method: str
    :param full_res: the dataframe with merged results
    :type full_res: pandas.DataFrame
    :return: the AUC for the given evaluation metrics
    """
    if eva_method == "diff":
        # calculate AUC for c-d model
        aucs = calculate_auc_for_diff_model(full_res["label"],
                                            full_res["con_perp"],
                                            full_res["dem_perp"])
    elif eva_method == "ratio":
        # calculate AUC for c/d model
        aucs = calculate_auc_for_ratio_model(full_res["label"],
                                             full_res["con_perp"],
                                             full_res["dem_perp"])
    else:
        aucs = calculate_auc_for_log_model(full_res["label"],
                                           full_res["con_perp"],
                                           full_res["dem_perp"])
    return aucs


def print_output(model_dem, eva_method):
    """
    print the training and test evaluation metrics for the model modifications

    :param model_dem: the modified model serving as dementia
    :type model_dem: transformers.modeling_gpt2.GPT2LMHeadModel
    :param eva_method: the evaluation method
    :return: a list of auc and accuracy on training and test set
    :rtype: list
    """
    # training transcript evaluation results
    train_res = evaluate_model_without_output(train_frame, model_dem, gpt_tokenizer)
    full_train_res = con_train_res.merge(train_res, on="file")
    full_train_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
    full_train_res = full_train_res.drop(["discard"], axis=1)
    # test transcript evaluation results
    test_res = evaluate_model_without_output(test_frame, model_dem, gpt_tokenizer)
    full_test_res = con_test_res.merge(test_res, on="file")
    full_test_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
    full_test_res = full_test_res.drop(["discard"], axis=1)
    # calcualte auc and accuracy for control model on training set
    labels = full_train_res["label"].values.tolist()
    con_perp = full_train_res["con_perp"].values.tolist()
    con_train_accu, con_train_auc = calcualte_accuracy(labels, con_perp)
    # calculate auc and accuracy for control model on test set
    labels = full_test_res["label"].values.tolist()
    con_perp = full_test_res["con_perp"].values.tolist()
    con_test_accu, con_test_auc = calcualte_accuracy(labels, con_perp)
    # calculate aucs and accuracies and return them
    train_accu, train_auc = calculate_aucs(eva_method, full_train_res)
    test_accu, test_auc = calculate_aucs(eva_method, full_test_res)
    results = [con_train_accu, con_train_auc, con_test_accu, con_test_auc,
               train_accu, train_auc, test_accu, test_auc]
    return results


def form_res_dict(model_modified, res_dict):
    """
    form the result dict with dementia model evaluation results

    :param model_modified: the modified GPT-2 model
    :type model_modified: transformers.modeling_gpt2.GPT2LMHeadModel
    :param res_dict: the dict for storing evaluation results
    :param res_dict: dict
    :return: the dict with all evaluation results
    """
    results = print_output(model_modified, "diff")
    results = [round(v, 3) for v in results]
    res_dict["train_con_accu"].append(results[0])
    res_dict["train_con_auc"].append(results[1])
    res_dict["test_con_accu"].append(results[2])
    res_dict["test_con_auc"].append(results[3])
    res_dict["train_diff_accu"].append(results[4])
    res_dict["train_diff_auc"].append(results[5])
    res_dict["test_diff_accu"].append(results[6])
    res_dict["test_diff_auc"].append(results[7])

    results = print_output(model_modified, "ratio")
    results = [round(v, 3) for v in results]
    res_dict["train_ratio_accu"].append(results[4])
    res_dict["train_ratio_auc"].append(results[5])
    res_dict["test_ratio_accu"].append(results[6])
    res_dict["test_ratio_auc"].append(results[7])

    results = print_output(model_modified, "log")
    results = [round(v, 3) for v in results]
    res_dict["train_log_accu"].append(results[4])
    res_dict["train_log_auc"].append(results[5])
    res_dict["test_log_accu"].append(results[6])
    res_dict["test_log_auc"].append(results[7])
    return res_dict


def print_dict_values(res_dict):
    """
    print out dict values for visualization

    :param res_dict: the dict with all evaluatoin results
    :type res_dict: dict
    """
    try:
        for k, v in res_dict.items():
            if len(v) > 0:
                sys.stdout.write("{} = {}\n".format(k, v))
    except AttributeError:
        pass


def onetime_train_process(share, generate_text):
    """
    zeroing certain share of attention heads in each layer,
    and just change one layer

    :param share: the % of attention heads to be zeroed
    :type share: int
    :param generate_text: indicator if text generation is needed
    :type generate_text: bool
    """
    res_dict = {"train_con_auc": [], "train_con_accu": [],
                "test_con_auc": [], "test_con_accu": [],
                "train_diff_auc": [], "train_diff_accu": [],
                "test_diff_auc": [], "test_diff_accu": [],
                "train_ratio_auc": [], "train_ratio_accu": [],
                "test_ratio_auc": [], "test_ratio_accu": [],
                "train_log_auc": [], "train_log_accu": [],
                "test_log_auc": [], "test_log_accu": []}
    style = "zero"
    for i in range(0, 12):
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        model_modified = break_attn_heads_by_layer(model_dem, share, i, style)
        if not generate_text:
            res_dict = form_res_dict(model_modified, res_dict)
        else:
            log_file = "logs/style_{}_share_{}_layer_{}.log".format(args.style, args.share, i)
            check_file(log_file)
            log = open(log_file, "a")
            warnings.filterwarnings("ignore")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                                filemode="a", level=logging.ERROR,
                                filename=log_file)
            sys.stdout = log
            output = generate_dem_text(model_modified, gpt_tokenizer)
            sys.stdout.write(output)
            sys.stdout.write("\n")
    print_dict_values(res_dict)


def accumu_train_process(share, num_layers, generate_text, res_dict):
    """
    accumulately zero attention heads
    zero layer 1, layer 1 & 2, layer 1 & 2 & 3, ... certain % attention heads

    :param share: the % of attention heads to be zeroed
    :type share: int
    :param num_layers: number of layers to be zeroed
    :param generate_text: the indicator if text generation is necessary
    :type generate_text: bool
    :param res_dict: the dict storing evaluation results
    :type res_dict: dict
    :return: the dict for storing evaluation results
    :rtype: dict
    """
    style = "zero"
    if num_layers > 13:
        raise ValueError("GPT-2 model only has 12 layers")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    for i in range(0, num_layers):
        model_dem = break_attn_heads_by_layer(model_dem, share, i, style)
    if generate_text:
        log_file = "logs/style_{}_share_{}_layer_{}.log".format(args.style, args.share, num_layers)
        check_file(log_file)
        log = open(log_file, "a")
        warnings.filterwarnings("ignore")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            filemode="a", level=logging.ERROR,
                            filename=log_file)
        sys.stdout = log
        sys.stdout.write(generate_dem_text(model_dem, gpt_tokenizer))
        sys.stdout.write("\n")
    else:
        res_dict = form_res_dict(model_dem, res_dict)
        return res_dict


def comb_train_process(share, generate_text):
    """
    zeoring the best combination of layers

    :param share: % of attention heads to be zeroed
    :type share: int
    :param generate_text: the indicator if the text generation is needed
    :type generate_text: bool
    """
    style = "zero"
    res_dict = {"train_con_auc": [], "train_con_accu": [],
                "test_con_auc": [], "test_con_accu": [],
                "train_diff_auc": [], "train_diff_accu": [],
                "test_diff_auc": [], "test_diff_accu": [],
                "train_ratio_auc": [], "train_ratio_accu": [],
                "test_ratio_auc": [], "test_ratio_accu": [],
                "train_log_auc": [], "train_log_accu": [],
                "test_log_auc": [], "test_log_accu": []}
    layers = [0, 1, 2, 3, 4, 8, 10]
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    for layer in layers:
        model_dem = break_attn_heads_by_layer(model_dem, share, layer, style)
    if generate_text:
        log_file = "logs/style_{}_share_{}.log".format(args.style, args.share)
        check_file(log_file)
        log = open(log_file, "a")
        warnings.filterwarnings("ignore")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            filemode="a", level=logging.ERROR,
                            filename=log_file)
        sys.stdout = log
        sys.stdout.write(generate_dem_text(model_dem, gpt_tokenizer))
        sys.stdout.write("\n")
    else:
        res_dict = form_res_dict(model_dem, res_dict)
    print_dict_values(res_dict)


if __name__ == "__main__":
    args = parse_args()
    '''
    LOG_FILE = "logs/zero_style_{}_eval_{}_share_{}.log".format(args.style, args.eval, args.share)
    log = open(LOG_FILE, "a")
    warnings.filterwarnings("ignore")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.ERROR,
                        filename=LOG_FILE)
    sys.stdout = log
    '''
    if args.style == "onetime":
        onetime_train_process(args.share, args.text)
    elif args.style == "accumu":
        res_dict = {"train_con_auc": [], "train_con_accu": [],
                    "test_con_auc": [], "test_con_accu": [],
                    "train_diff_auc": [], "train_diff_accu": [],
                    "test_diff_auc": [], "test_diff_accu": [],
                    "train_ratio_auc": [], "train_ratio_accu": [],
                    "test_ratio_auc": [], "test_ratio_accu": [],
                    "train_log_auc": [], "train_log_accu": [],
                    "test_log_auc": [], "test_log_accu": []}
        for accu_layer in range(1, 13):
            res_dict = accumu_train_process(args.share, accu_layer, args.text, res_dict)
        print_dict_values(res_dict)
    elif args.style == "comb":
        comb_train_process(args.share, args.text)
    else:
        raise ValueError("method not supported!")
