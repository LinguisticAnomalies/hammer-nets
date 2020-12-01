"""
Zeroing attention heads in each layer, in terms of ont-time or accumulately.
"""


import sys
import argparse
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
    return parser.parse_args()


def calculate_aucs(eva_method, full_res):
    """
    calcualte different aucs with given folder prefix and evaluation method

    :param eva_method: the evaluation method
    :type eva_method: str
    :param full_res: the dataframe with merged results
    :type full_res: pandas.DataFrame
    :return: the AUC for c-d, c/d or log(c) - log(d)
    """
    if eva_method == "diff":
        # calculate AUC for c-d model
        diff_auc = calculate_auc_for_diff_model(full_res["label"],
                                                full_res["con_perp"],
                                                full_res["dem_perp"])
        return diff_auc
    elif eva_method == "ratio":
        # calculate AUC for c/d model
        ratio_auc = calculate_auc_for_ratio_model(full_res["label"],
                                                  full_res["con_perp"],
                                                  full_res["dem_perp"])
        return ratio_auc
    elif eva_method == "log":
        log_auc = calculate_auc_for_log_model(full_res["label"],
                                              full_res["con_perp"],
                                              full_res["dem_perp"])
        return log_auc


def print_output(model_dem):
    """
    print the training and test evaluation metrics for the model modifications

    :param model_dem: the modified model serving as dementia
    :type model_dem: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    eval_methods = ["diff", "ratio", "log"]
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
    # calculate aucs and print them out
    for item in eval_methods:
        train_accu, train_auc = calculate_aucs(item, full_train_res)
        sys.stdout.write("training set in {} auc: {}\n".format(item, round(train_auc, 3)))
        sys.stdout.write("training set in {} accuracy: {}\n".format(item, round(train_accu, 3)))
        test_accu, test_auc = calculate_aucs(item, full_test_res)
        sys.stdout.write("test set in {} auc: {}\n".format(item, round(test_auc, 3)))
        sys.stdout.write("test set in {} accuracy: {}\n".format(item, round(test_accu, 3)))



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
    for i in range(0, 12):
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        model_modified = break_attn_heads_by_layer(model_dem, share, i, style)
        dem_res = evaluate_model_without_output(train_frame, model_modified, gpt_tokenizer)
        full_res = con_train_res.merge(dem_res, on="file")
        full_res.columns = ["file", "label", "con_perp", "discard", "dem_perp"]
        full_res = full_res.drop(["discard"], axis=1)
        calculate_aucs(eva_method, full_res)
        '''
        out_file = "layer_{}_dem.json".format(i)
        evaluate_model_with_output(test_frame, model_modified, gpt_tokenizer,
                                   folder_prefix, out_file)
        '''
        generate_dem_text(model_dem, gpt_tokenizer)
        

def accumu_train_process(share, num_layers):
    """
    accumulately zero attention heads
    zero layer 1, layer 1 & 2, layer 1 & 2 & 3, ... certain % attention heads

    :param share: the % of attention heads to be zeroed
    :type share: int
    :param num_layers: number of layers to be zeroed
    """
    style = "zero"
    if num_layers > 13:
        raise ValueError("GPT-2 model only has 12 layers")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    for i in range(0, num_layers):
        model_dem = break_attn_heads_by_layer(model_dem, share, i, style)
    sys.stdout.write("="*20)
    sys.stdout.write("\n")
    sys.stdout.write("first {} layers\n".format(num_layers))
    print_output(model_dem)
    generate_dem_text(model_dem, gpt_tokenizer)
    sys.stdout.write("="*20)
    sys.stdout.write("\n")


def comb_train_process(share):
    """
    zeoring the best combination of layers

    :param share: [description]
    :type share: [type]
    """
    style = "zero"
    layers = [0, 1, 2, 3, 4, 8, 10]
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    for layer in layers:
        model_dem = break_attn_heads_by_layer(model_dem, share, layer, style)
    print_output(model_dem)
    generate_dem_text(model_dem, gpt_tokenizer)


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
        onetime_train_process(args.share)
    elif args.style == "accumu":
        for accu_layer in range(1, 13):
            accumu_train_process(args.share, accu_layer)
    elif args.style == "comb":
        comb_train_process(args.share)
    else:
        raise ValueError("method not supported!")