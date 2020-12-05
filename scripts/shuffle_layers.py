"""
Breaking the nerual network by shuffling attention heads for each layer
with 100 epochs
    - Test for reproducibility
    - Find the model with best c-d, c/d, log(c)-log(d) AUC
    - Calcualte bootstrap confidence interval
    - Calculate the accuracy for the best performing model
Shuffle an attention vector 10 times, re-run the script for 10 times as 100 epochs
"""

import os
import sys
import argparse
import warnings
import logging
import pickle
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
    add shuffling head argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int,
                        help="the n-th layer to be shuffled")
    parser.add_argument("--share", type=int,
                        help="the % of attention heads to be modified")
    parser.add_argument("--eval", type=str,
                        help="the evaluation metrics")
    parser.add_argument("--batch", type=int,
                        help="the current batch for epoch testing")
    return parser.parse_args()


def train_process_for_shuffle(nth_layer, num_epochs,
                              share, eva_method):
    """
    The training process for shuffling 25%, 50% and 100% attention heads in each layer.
    The layer modeficiation is one-time, and the modified model will be evaluated in
    different metrics: c-d, c/d, and log(c) - log(d)

    :param nth_layer: the n-th layer to be modified, ranging from 0 to 11
    :type nth_head: int
    :param num_epochs: number of epochs for reproducible tesing
    :type num_epochs: int
    :param share: the % of attention heads to be modified, ranging from 25, 50, 100
    :type share: int
    :param eva_method: the methods for evaluation, including 'diff', 'ratio' and 'log'

    :return: the best AUC over num_epochs
    :rtype: float
    :return: the best c-d model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    style = "shuffle"
    best_model = None
    best_auc = 0.0
    pkl_prefix = "pkls/style-{}-eval-{}/".format(style, eva_method)
    pkl_file = "layer-{}-share-{}.pkl".format(nth_layer,
                                              share)
    # create the sub-folder if not exists
    if not os.path.isdir(pkl_prefix):
        os.mkdir(pkl_prefix)
    # load the best auc if it is not the first batch
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as auc_f:
            epoch_aucs = pickle.load(auc_f)
        best_auc = max(epoch_aucs)
    else:
        epoch_aucs = []
    for i in range(0, num_epochs):
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        sys.stdout.write("\tepochs {}...\n".format(i+1))
        model_modified = break_attn_heads_by_layer(model_dem, share, nth_layer, style)
        dem_res = evaluate_model_without_output(train_frame, model_modified, gpt_tokenizer)
        full_res = con_train_res.merge(dem_res, on="file")
        full_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
        full_res = full_res.drop(["discard"], axis=1)
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
        elif eva_method == "log":
            aucs = calculate_auc_for_log_model(full_res["label"],
                                               full_res["con_perp"],
                                               full_res["dem_perp"])
        else:
            raise ValueError("evaluation metrics not supported!")
        epoch_aucs.append(aucs)
        if aucs > best_auc:
            best_auc = aucs
            best_model = model_modified
        with open(pkl_prefix + pkl_file, "wb") as auc_f:
            pickle.dump(epoch_aucs, auc_f)
    return best_auc, best_model


def epoch_main_process(layer, share, epochs, eva_method):
    """
    the main function for epoch training
    shuffle GPT-2 model with certain number of epochs,
    shuffle one attention vector a time
    evaluate the best model from num_epochs and training set and evaluate it on test set,
    save the result to local file

    :param layer: the n-th layer to be modified
    :type layer: int
    :param share: the % of attention heads to be changed
    :type share: int
    :param epochs: number of epochs for testing
    :type epochs: int
    :param eva_method: the evaluation method
    :type eva_method: str
    """
    folder_prefix = "../results/onetime-share-{}-style-shuffle-eval-{}/".format(share,
                                                                                eva_method)
    if not os.path.isdir(folder_prefix):
        os.mkdir(folder_prefix)
    best_auc, best_model = train_process_for_shuffle(layer, epochs,
                                                     share, eva_method)
    if best_model:
        out_file = "layer_{}_dem.json".format(layer)
        sys.stdout.write("evaluation method: {}\n".format(eva_method))
        sys.stdout.write("best auc for current batch: {}\n".format(best_auc))
        evaluate_model_with_output(test_frame, best_model, gpt_tokenizer, folder_prefix, out_file)
        generate_dem_text(share, layer, best_model, gpt_tokenizer)


if __name__ == "__main__":
    BATCH_EPOCHS = 20
    args = parse_args()
    LOG_FILE = "logs/onetime_layer_{}_style_shuffle_share_{}_eva_{}.log".format(args.layer,
                                                                                args.share,
                                                                                args.eval)
    log = open(LOG_FILE, "a")
    warnings.filterwarnings("ignore")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.ERROR,
                        filename=LOG_FILE)
    sys.stdout = log
    epoch_main_process(args.layer, args.share, BATCH_EPOCHS, args.eval)
