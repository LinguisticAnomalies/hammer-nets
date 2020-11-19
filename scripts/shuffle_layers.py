"""
Breaking the nerual network by shuffling attention heads for each layer with a large number of epochs
    - Test for reproducibility
    - Find the model with best c-d AUC
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
from util_fun import calculate_auc_for_diff_model, evaluate_model_with_output, break_attn_heads_by_layer


gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
train_frame = pd.read_csv("address_train.csv")
test_frame = pd.read_csv("address_test.csv")


def parse_args():
    """
    add shuffling head argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int,
                        help="the n-th layer to be shuffled")
    parser.add_argument("--share", type=int,
                        help="the % of attention heads to be modified")
    parser.add_argument("--batch", type=int,
                        help="the current batch for epoch testing")
    return parser.parse_args()


def train_process_for_shuffle(eva_frame, num_layer, num_epochs, con_train_res, share, style):
    """
    the epoch training process for shuffling ttention heads of GPT-2 model,
    evaluate on the ADDRESS training set
    return the model with the best c-d AUC

    :param eva_frame: the training transcript dataframe to be evaluated
    :type eva_frame: pandas.DataFrame
    :param num_layer: the n-th layer to be modified, ranging from 0 to 11
    :type num_head: int
    :param num_epochs: number of epochs for reproducible tesing
    :type num_epochs: int
    :param con_train_res: the evaluation result from control model on the training set
    :type con_train_res: pandas.DataFrame
    :param share: the % of attention heads to be modified, ranging from 25, 50, 100
    :type share: int
    :param style: the modifying style, choose from 'zero' or 'style'
    :type stype: str
    :return: the best AUC for c-d model over num_epochs
    :rtype: float
    :return: the best c-d model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    best_model = None
    best_auc = 0.0
    # save all epoch auc to a local pickle file
    pkl_file = "pkls/" + str(style) + "/layer_" + str(num_layer) + "_" + \
        str(num_epochs) + "_epochs_" + str(share) + "_share.pkl"
    # if not the first 10 epochs
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as auc_f:
            epoch_aucs = pickle.load(auc_f)
        best_auc = max(epoch_aucs)
    else:
        epoch_aucs = []
    for i in range(0, num_epochs):
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        sys.stdout.write("\tepochs {}...\n".format(i+1))
        model_modified = break_attn_heads_by_layer(model_dem, share, num_layer, style)
        dem_res = evaluate_model_without_output(eva_frame, model_modified, gpt_tokenizer)
        full_res = con_train_res.merge(dem_res, on="file")
        full_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
        full_res = full_res.drop(["discard"], axis=1)
        # calculate AUC for c-d model
        diff_auc = calculate_auc_for_diff_model(full_res["label"],
                                                full_res["con_perp"],
                                                full_res["dem_perp"])
        epoch_aucs.append(diff_auc)
        if diff_auc > best_auc:
            best_auc = diff_auc
            best_model = model_modified
    with open(pkl_file, "wb") as auc_f:
        pickle.dump(epoch_aucs, auc_f)
    return best_model


def epoch_main_process(layer, share, style, epochs):
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
    """
    con_train_res = read_json("../results/cache-original/con_train.json")
    folder_prefix = "../results/cache-epochs-100-share-" + str(share) + "-style-" + str(style) + "/"
    best_model = train_process_for_shuffle(train_frame, layer,
                                            epochs, con_train_res,
                                            share, style)
    # check if the current batch has the best model,
    # if so, evaluate the best model with test set and save the result to local file
    # otherwise, continue to next batch
    if best_model:
        out_file = "layer_" + str(layer) + "_" + \
            str(share) + "_share_" + str(style) + "_gpt2_dem.json"
        # save the generated text from the best best model to log file
        evaluate_model_with_output(test_frame, best_model, gpt_tokenizer, folder_prefix, out_file)
        generate_dem_text(share, layer, best_model, gpt_tokenizer)


if __name__ == "__main__":
    epochs = 1
    style = "zero"
    args = parse_args()
    log_file = "logs/epoch_" + str(epochs) + "_batch_" + str(args.batch) + \
        "_layer_" + style + "_" + \
            str(args.share) + "_share_" + style + ".log"
    if os.path.exists(log_file):
        commend = "rm " + log_file
        os.system(commend)
    log = open(log_file, "a")
    warnings.filterwarnings("ignore")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.ERROR,
                        filename=log_file)
    sys.stdout = log
    epoch_main_process(args.layer, args.share, style, epochs)