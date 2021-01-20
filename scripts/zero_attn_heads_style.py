'''
Rewrite zero_attn_heads.py file with more flexibility
'''

import sys
import pickle
from datetime import datetime
import gc
import os
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import calculate_metrics, break_attn_heads_by_layer, str2bool, generate_texts
# use GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    """
    add zeroing head argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hammer_style", type=str,
                        help="the style of changing attention heads, including oneime, accumu and combo")
    parser.add_argument("--zero_style", type=str,
                        help="the style of zeroing attn heads, supporting 'random','first' and 'shuffle'")
    parser.add_argument("--data_type", type=str,
                        help="the type of data to load, supporting 'full', 'mild' and 'slight'")
    parser.add_argument("--share", type=int,
                        help="the % attention heads to be changed")
    parser.add_argument('--text', type=str2bool,
                        help="boolean indicator for text generation with dementia model")
    return parser.parse_args()


def get_data_name(data_type):
    """
    given the data type, return dataset names

    :param data_type: the dataset type
    :type data_type: str
    """
    if data_type == "full":
        train_data = "address_train_full"
        test_data = "address_test_full"
    elif data_type == "mild":
        train_data = "address_train_mild"
        test_data = "address_test_mild"
    elif data_type == "slight":
        train_data = "address_train_slight"
        test_data = "address_test_slight"
    else:
        raise ValueError("dataset type not supported!")
    return train_data, test_data


def onetime_train_process(data_type, zero_style, share, text_generate=False):
    """
    zeroing certain share of attention heads with specific style in each layer,
    evaluate the model after the change,
    save the evaluation metrics to local file

    :param data_type: full, mild or slight address dataset
    :param zero_style: the style of zeroing attention heads 
    :type zero_style: str
    :param share: % of attention heads to be zeroed
    :type share: int
    :param text_generate: the indicator for text generation, defaults to False
    :type text_generate: bool, optional
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    # TODO: only supports ADreSS dataset for now
    train_data, test_data = get_data_name(data_type)
    res_dict = {"train_con_auc": [], "train_con_accu": [],
                "test_con_auc": [], "test_con_accu": [],
                "train_diff_auc": [], "train_diff_accu": [],
                "test_diff_auc": [], "test_diff_accu": [],
                "train_ratio_auc": [], "train_ratio_accu": [],
                "test_ratio_auc": [], "test_ratio_accu": [],
                "train_log_auc": [], "train_log_accu": [],
                "test_log_auc": [], "test_log_accu": []}
    for i in range(0, 12):
        # sys.stdout.write("onetime zeroing {} {}% attn heads on layer {}\n".format(zero_style, share, i))
        model_con = GPT2LMHeadModel.from_pretrained("gpt2")
        model_dem = break_attn_heads_by_layer(zero_style, model_con, share, i)
        if text_generate:
            out_file = "../results/text/onetime_{}_{}_{}_layer_{}.tsv".format(zero_style, share, data_type, i)
            generate_texts(model_con, model_dem, gpt_tokenizer, out_file)
        else:
            res_dict = calculate_metrics(res_dict, model_dem, gpt_tokenizer, train_data, test_data)
        del model_dem, model_con
        gc.collect()
    # save evaluation metrics to local pickle file
    pickle_file = "../results/evals/onetime_{}_{}_{}.pkl".format(zero_style, share, data_type)
    with open(pickle_file, "wb") as f:
        pickle.dump(res_dict, f)
    del gpt_tokenizer, res_dict
    gc.collect()


def accumu_model_driver(model, share, zero_style, num_layers):
    """
    the driver function for breaking GPT-2 model

    :param model: the oringal GPT-2 model to be modified
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param share: % of attention heads to be zeroed
    :type share: int
    :param zero_style: the style of zeroing attention heads 
    :type zero_style: str
    :param num_layers: numer of layers to be zeroed
    :type num_layers: int
    :return: the modified model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    if num_layers > 13:
        raise ValueError("GPT-2 model only has 12 layers")
    for i in range(0, num_layers):
        # sys.stdout.write("accumu zeroing {} {}% attn heads on first {} layer(s)\n".format(zero_style, share, i))
        # be aware that zeroing the first layer = zeroing 0th layer in GPT-2 model
        model = break_attn_heads_by_layer(zero_style, model, share, i)
    return model


def accumu_train_process(data_type, zero_style, share, text_generate=False):
    """
    zeroing certain share of attn heads with specific style accumulately,
    zeroing layer 1, layer 1 & 2 and layer 1 & 2 & 3, ... with certain % attn heads

    :param data_type: full, mild or slight address dataset
    :param zero_style: the style of zeroing attention heads 
    :type zero_style: str
    :param share: % of attention heads to be zeroed
    :type share: int
    :param text_generate: the indicator for text generation, defaults to False
    :type text_generate: bool, optional
    """
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    train_data, test_data = get_data_name(data_type)
    res_dict = {"train_con_auc": [], "train_con_accu": [],
                "test_con_auc": [], "test_con_accu": [],
                "train_diff_auc": [], "train_diff_accu": [],
                "test_diff_auc": [], "test_diff_accu": [],
                "train_ratio_auc": [], "train_ratio_accu": [],
                "test_ratio_auc": [], "test_ratio_accu": [],
                "train_log_auc": [], "train_log_accu": [],
                "test_log_auc": [], "test_log_accu": []}
    for accu_layer in range(1, 13):
        model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
        model_dem = accumu_model_driver(model_dem, share, zero_style, accu_layer)
        if text_generate:
            out_file = "../results/text/accumu_{}_{}_{}_layer_{}.tsv".format(zero_style, share, data_type, accu_layer)
            generate_texts(model_con, model_dem, gpt_tokenizer, out_file)
        else:
            res_dict = calculate_metrics(res_dict, model_dem, gpt_tokenizer, train_data, test_data)
        del model_dem
        gc.collect()
        # save evaluation metrics to local pickle file
    pickle_file = "../results/evals/accumu_{}_{}_{}.pkl".format(zero_style, share, data_type)
    with open(pickle_file, "wb") as f:
        pickle.dump(res_dict, f)
    del gpt_tokenizer, res_dict, model_con
    gc.collect()


def combo_train_process(data_type, zero_style, share, text_generate=False):
    """
    zeroing attn heads with certain style, share and best layer combination

    :param data_type: full, mild or slight address dataset
    :param zero_style: the style of zeroing attention heads 
    :type zero_style: str
    :param share: % of attention heads to be zeroed
    :type share: int
    :param text_generate: the indicator for text generation, defaults to False
    :type text_generate: bool, optional
    """
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    res_dict = {"train_con_auc": [], "train_con_accu": [],
                "test_con_auc": [], "test_con_accu": [],
                "train_diff_auc": [], "train_diff_accu": [],
                "test_diff_auc": [], "test_diff_accu": [],
                "train_ratio_auc": [], "train_ratio_accu": [],
                "test_ratio_auc": [], "test_ratio_accu": [],
                "train_log_auc": [], "train_log_accu": [],
                "test_log_auc": [], "test_log_accu": []}
    train_data, test_data = get_data_name(data_type)
    layers = [0, 1, 2, 3, 4, 8, 10]
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    for layer in layers:
        # sys.stdout.write("combo zeroing {} {}% attn heads on layer {}\n".format(zero_style, share, layer))
        model_dem = break_attn_heads_by_layer(zero_style, model_dem, share, layer)
    if text_generate:
        out_file = "../results/text/comb_{}_{}_{}.tsv".format(zero_style, share, data_type)
        generate_texts(model_con, model_dem, gpt_tokenizer, out_file)
    else:
        res_dict = calculate_metrics(res_dict, model_dem, gpt_tokenizer, train_data, test_data)
    pickle_file = "../results/evals/comb_{}_{}_{}.pkl".format(zero_style, share, data_type)
    with open(pickle_file, "wb") as f:
        pickle.dump(res_dict, f)
    del gpt_tokenizer, res_dict, model_dem
    gc.collect()


if __name__ == "__main__":
    start_time = datetime.now()
    args = parse_args()
    if args.hammer_style == "onetime":
        onetime_train_process(args.data_type, args.zero_style, args.share, args.text)
    elif args.hammer_style == "accumu":
        accumu_train_process(args.data_type, args.zero_style, args.share, args.text)
    elif args.hammer_style == "comb":
        combo_train_process(args.data_type, args.zero_style, args.share, args.text)
    else:
        raise ValueError("method not supported")
    sys.stdout.write("Total time running :{}\n".format(datetime.now() - start_time))