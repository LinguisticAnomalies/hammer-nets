"""
Zeroing attention heads in each layer, in terms of ont-time or accumulately.
"""


import os
import sys
import argparse
import warnings
import logging
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model_without_output, generate_dem_text, read_json
from util_fun import calculate_auc_for_diff_model, calculate_auc_for_ratio_model
from util_fun import calculate_auc_for_log_model
from util_fun import evaluate_model_with_output, break_attn_heads_by_layer


gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
train_frame = pd.read_csv("address_train.csv")
test_frame = pd.read_csv("address_test.csv")
con_train_res = read_json("../results/cache-original/con_train.json")


def parse_args():
    """
    add zeroing head argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str,
                        help="the style of changing attention heads")
    parser.add_argument("--share", type=int,
                        help="the % of attention heads to be modified")
    parser.add_argument("--eval", type=str,
                        help="the evaluation metrics")
    return parser.parse_args()


def calculate_aucs(eva_method, full_res):
    """
    calcualte different aucs with given folder prefix and evaluation method

    :param eva_method: the evaluation method
    :type eva_method: str
    :param full_res: the dataframe with merged results
    :type full_res: pandas.DataFrame
    """
    if eva_method == "diff":
        # calculate AUC for c-d model
        diff_auc = calculate_auc_for_diff_model(full_res["label"],
                                                full_res["con_perp"],
                                                full_res["dem_perp"])
        sys.stdout.write("c-d model AUC: \t{}\n".format(diff_auc))
    elif eva_method == "ratio":
        # calculate AUC for c/d model
        ratio_auc = calculate_auc_for_ratio_model(full_res["label"],
                                                  full_res["con_perp"],
                                                  full_res["dem_perp"])
        sys.stdout.write("c/d model AUC: \t{}\n".format(ratio_auc))
    elif eva_method == "log":
        log_auc = calculate_auc_for_log_model(full_res["label"],
                                              full_res["con_perp"],
                                              full_res["dem_perp"])
        sys.stdout.write("log(c)-log(d) model AUC:\t{}\n".format(log_auc))


def onetime_train_process(share, eva_method):
    """
    zeroing certain share of attention heads in each layer,
    and just change one layer

    :param share: the % of attention heads to be zeroed
    :type share: int
    :param eva_method: the evaluation metrics
    :type eva_method: str
    """
    style = "zero"
    folder_prefix = "../results/onetime-share-{}-style-{}-eval-{}/".format(share, style,
                                                                           eva_method)
    if not os.path.isdir(folder_prefix):
        os.mkdir(folder_prefix)
    for i in range(0, 12):
        sys.stdout.write("===========================\n")
        sys.stdout.write("zeroing layer {} {}% attention heads\n".format(i, share))
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        model_modified = break_attn_heads_by_layer(model_dem, share, i, style)
        dem_res = evaluate_model_without_output(train_frame, model_modified, gpt_tokenizer)
        full_res = con_train_res.merge(dem_res, on="file")
        full_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
        full_res = full_res.drop(["discard"], axis=1)
        calculate_aucs(eva_method, full_res)
        out_file = "layer_{}_dem.json".format(i)
        evaluate_model_with_output(test_frame, model_modified, gpt_tokenizer,
                                   folder_prefix, out_file)
        sys.stdout.write("---------------------------\n")
        generate_dem_text(share, i, model_modified, gpt_tokenizer)
        sys.stdout.write("---------------------------\n")
        sys.stdout.write("===========================\n")


def accumu_train_process(share, eva_method, num_layers):
    """
    accumulately zero attention heads
    zero layer 1, layer 1 & 2, layer 1 & 2 & 3, ... certain % attention heads

    :param share: the % of attention heads to be zeroed
    :type share: int
    :param eva_method: the evaluation metrics
    :type eva_method: str
    :param num_layers: number of layers to be zeroed
    """
    style = "zero"
    folder_prefix = "../results/accumu-share-{}-style-{}-eval-{}/".format(share, style,
                                                                          eva_method)
    if not os.path.isdir(folder_prefix):
        os.mkdir(folder_prefix)
    if num_layers > 13:
        raise ValueError("GPT-2 model only has 12 layers")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    sys.stdout.write("===========================\n")
    sys.stdout.write("zeroing first {} layer(s) {}% attention heads\n".format(num_layers, share))
    for i in range(0, num_layers):
        model_dem = break_attn_heads_by_layer(model_dem, share, i, style)
    dem_res = evaluate_model_without_output(train_frame, model_dem, gpt_tokenizer)
    full_res = con_train_res.merge(dem_res, on="file")
    full_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
    full_res = full_res.drop(["discard"], axis=1)
    calculate_aucs(eva_method, full_res)
    out_file = "layer_{}_dem.json".format(num_layers)
    evaluate_model_with_output(test_frame, model_dem, gpt_tokenizer,
                               folder_prefix, out_file)
    sys.stdout.write("---------------------------\n")
    generate_dem_text(share, num_layers, model_dem, gpt_tokenizer)
    sys.stdout.write("---------------------------\n")
    sys.stdout.write("===========================\n")


def comb_train_process(share, eva_method):
    """
    zeoring the best combination of layers

    :param share: [description]
    :type share: [type]
    :param eva_method: [description]
    :type eva_method: [type]
    """
    style = "zero"
    folder_prefix = "../results/comb-share-{}-style-{}-eval-{}/".format(share, style,
                                                                        eva_method)
    if not os.path.isdir(folder_prefix):
        os.mkdir(folder_prefix)
    layers = [0, 1, 2, 3, 4, 8, 10]
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    for layer in layers:
        model_dem = break_attn_heads_by_layer(model_dem, share, layer, style)
    dem_res = evaluate_model_without_output(train_frame, model_dem, gpt_tokenizer)
    full_res = con_train_res.merge(dem_res, on="file")
    full_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
    full_res = full_res.drop(["discard"], axis=1)
    calculate_aucs(eva_method, full_res)
    out_file = "layer_dem.json"
    evaluate_model_with_output(test_frame, model_dem, gpt_tokenizer,
                               folder_prefix, out_file)
    sys.stdout.write("---------------------------\n")
    generate_dem_text(share, len(layers), model_dem, gpt_tokenizer)
    sys.stdout.write("---------------------------\n")
    sys.stdout.write("===========================\n")


if __name__ == "__main__":
    args = parse_args()
    LOG_FILE = "logs/zero_style_{}_eval_{}_share_{}.log".format(args.style, args.eval, args.share)
    log = open(LOG_FILE, "a")
    warnings.filterwarnings("ignore")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.ERROR,
                        filename=LOG_FILE)
    sys.stdout = log
    if args.style == "onetime":
        onetime_train_process(args.share, args.eval)
    elif args.style == "accumu":
        for accu_layer in range(1, 13):
            accumu_train_process(args.share, args.eval, accu_layer)
    elif args.style == "comb":
        comb_train_process(args.share, args.eval)
    else:
        raise ValueError("method not supported!")