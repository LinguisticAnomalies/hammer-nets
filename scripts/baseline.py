'''
This script is to fine-tune Longformer model and cross-validate on dataset from another domain/task
'''
import os
import argparse
import sys
import logging
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import LongformerForSequenceClassification
from transformers import LongformerTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import auc, roc_curve, accuracy_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
EPOCHS = 10
NUM_LABELS = 2
BATCH_SIZE = 8


def parse_args():
    """
    add arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="""the choice of model""")
    return parser.parse_args()



def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss

    :param elapsed: running time in second
    :type elapsed: float
    :return: running time in hh:mm:ss format
    :rtype: str
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seed(i):
    """
    set all possible seed to RANDOM_SEED

    :param i: the i-th run
    :type i: int
    """
    torch.manual_seed(RANDOM_SEED+i)
    np.random.seed(RANDOM_SEED+i)
    torch.cuda.manual_seed_all(RANDOM_SEED+i)


def calculate_flat_rate(labels, preds):
    """
    calcualte AUC and accuracy at flat rate,
    return AUC and accuracy

    :param labels: a list of true label
    :type labels: list
    :param preds: a list of prediction
    :type preds: list
    :return: the ACC & AUC at flat level
    :rtype: float
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    flat_auc = auc(fpr, tpr)
    flat_acc = accuracy_score(labels, preds)
    pred_df = pd.DataFrame({"pred": preds, "label": labels})
    pred_df.to_csv("pred_df.csv", index=False)
    return flat_acc, flat_auc


def calculate_eer_rate(labels, probs):
    """
    calculate AUC and accuracy at EER

    :param labels: a list of true label
    :type labels: list
    :param probs: a list of probalities represents the probability of being 1s
    :type preds: list
    :return: the ACC & AUC at flat level
    :rtype: float
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    eer_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    tnr = 1 - fpr
    prevalence = np.count_nonzero(labels)/len(labels)
    eer_point = np.nanargmin(np.absolute((fnr - fpr)))
    tpr_at_eer = tpr[eer_point]
    tnr_at_eer = tnr[eer_point]
    # flat accuracy and AUC
    eer_acc = tpr_at_eer * prevalence + tnr_at_eer * (1-prevalence)
    return eer_acc, eer_auc


def format_input(trans, tokenizer, max_len):
    """
    format the input sequence,
    return input ids and attention masks as tensors

    :param trans: a list of transcripts
    :type trans: list
    param tokenizer: bert tokenizer
    :type tokenizer: transformers.tokenization_bert.BertTokenizer
    :param max_len: the maximum padded length for the input sequence
    :return: input ids and attention masks
    :rtype: torch.Tensor
    """
    input_ids = []
    attn_masks = []
    for tran in trans:
        encoded_dict = tokenizer.encode_plus(
            tran, add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True)
        input_ids.append(encoded_dict["input_ids"])
        attn_masks.append(encoded_dict["attention_mask"])
    # conver the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attn_masks = torch.cat(attn_masks, dim=0)
    return input_ids, attn_masks


def set_dataloader(input_ids, attn_masks, label):
    """
    divide the formatted inputs into 90% training and 10% validation set,
    return the defined data loader object

    :param input_ids: the formatted input ids from transcripts
    :type input_ids: torch.Tensor
    :param attn_masks: the formatted attention masks from transcripts
    :type attn_masks: torch.Tensor
    :param label: the corresponding label from transcripts
    :type label: torch.Tensor
    :return: the defined dataloader
    :rtype: torch.utils.data.dataloader.DataLoader
    """
    dataset = TensorDataset(input_ids, attn_masks, label)
    df_dataloader = DataLoader(
        dataset, sampler=SequentialSampler(dataset),
        batch_size=BATCH_SIZE)
    return df_dataloader


def fine_tune_process(data_loader, model, save_name):
    """
    the actual fine tuning process,
    return the fine-tuned model

    :param data_loader: the data loader from training set
    :type data_loader: torch.utils.data.dataloader.DataLoader
    :param model: the fine-tuned model
    :type model: transformers.AutoModelForSequenceClassification
    :param save_name: the relative name to save the fine-tuned model
    :type save_name: str
    """
    # to use the second GPU
    # torch.cuda.set_device(1)
    total_t0 = time.time()
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(data_loader)*EPOCHS)
    for epoch_i in range(0, EPOCHS):
        #########################################
        # training
        #########################################
        model.train()
        set_seed(epoch_i)
        sys.stdout.write("\n")
        sys.stdout.write(
            '======== Epoch {:} / {:} ========\n'.format(
                epoch_i + 1, EPOCHS))
        sys.stdout.write('Training...\n')
        t0 = time.time()
        total_loss = 0.0
        for step, batch in enumerate(data_loader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                sys.stdout.write(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.\n'.format(
                    step, len(data_loader), elapsed))
            # send to cuda
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)
            # clear previously calculated gradients
            model.zero_grad()
            # forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels)
            loss = outputs[0]
            # add trainign loss
            total_loss += loss.item()
            # backward pass
            loss.backward()
            # prevent exploding gradients problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # update parameters
            optimizer.step()
            scheduler.step()
        avg_loss = total_loss/len(data_loader)
        run_time = format_time(time.time() - t0)
        sys.stdout.write("\n")
        sys.stdout.write(
            "  Average training loss: {0:.2f}\n".format(avg_loss))
        sys.stdout.write(
            "  Training epcoh took: {:}\n".format(run_time))
    sys.stdout.write(
        "Total training took {:} (h:mm:ss)\n".format(
            format_time(time.time()-total_t0)))
    sys.stdout.write("Training complete!\n")
    # check if fine_tuned folder exists
    sys.stdout.write("Saving model as {}\n".format(save_name))
    if not os.path.isdir("../fine_tuned/"):
        os.mkdir("../fine_tuned/")
    model.save_pretrained(os.path.join("../fine_tuned/", save_name))


def val_process(val_dataloader, model_ft):
    """
    the validation process,
    calculate accuracy and AUC at equal error rate

    :param val_dataloader: the validation data loader
    :type val_dataloader: torch.utils.data.dataloader.DataLoader
    :param model_ft: the fine-tuned model
    :type model: transformers.AutoModelForSequenceClassification
    """
    ######################################
    # validation
    ######################################
    preds = []
    labels = []
    probs = []
    sys.stdout.write("\n")
    sys.stdout.write("Running Validation...\n")
    t0 = time.time()
    # set to prediciton mode
    model_ft.eval()
    total_eval_loss = 0.0
    for batch in val_dataloader:
        b_input_ids = batch[0].to(DEVICE)
        b_input_mask = batch[1].to(DEVICE)
        b_labels = batch[2].to(DEVICE)
        # no forward pass
        with torch.no_grad():
            outputs = model_ft(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            total_eval_loss += loss.item()
            # move things to CPU for accuracy calculation
            prob = F.softmax(logits.detach().cpu(), dim=-1)
            # get the second column from probablities for metrics@EER
            for v in prob:
                probs.append(v.data[1].numpy())
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            # get the actual preditions
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            # add to the total list
            preds.extend(pred_flat)
            labels.extend(labels_flat)
    # accuracy and AUC at equal error rate
    eer_acc, eer_auc = calculate_eer_rate(labels, probs)
    flat_acc, flat_auc = calculate_flat_rate(labels, preds, )
    sys.stdout.write(
        "  Accuracy@flat: {0:.2f}\n".format(flat_acc))
    sys.stdout.write(
        "  AUC@flat: {0:.2f}\n".format(flat_auc))
    sys.stdout.write("--------\n")
    sys.stdout.write(
        "  Accuracy@EER: {0:.2f}\n".format(eer_acc))
    sys.stdout.write(
        "  AUC@EER: {0:.2f}\n".format(eer_auc))
    # calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss/len(val_dataloader)
    # measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    sys.stdout.write(
        "  Validation Loss: {0:.2f}\n".format(
            avg_val_loss))
    sys.stdout.write(
        "  Validation took: {:}\n".format(
            validation_time))
    sys.stdout.write("\n")
    sys.stdout.write("Validation complete!\n")


def fine_tune_driver(df_to_train, df_to_val, model_chose, ft_model_name):
    """
    fine tune Longformer on df_to_train and evalute on df_to_val,
    ignore all characters after 4,096

    :param df_to_train: the dataframe to be fine-tuned
    :type df_to_train: pd.DataFrame
    :param df_to_val: the dataframe to be validated
    :type df_to_val: pd.DataFrame
    :param model_chose: the model choise
    :type model_chose: str
    :param ft_model_name: the name of fine-tuned model
    :type: str
    """
    torch.device(DEVICE)
    if model_chose == "bert":
        sys.stdout.write("Loading BERT base model...\n")
        max_len = 512
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = NUM_LABELS,
            output_attentions = False,
            output_hidden_states = False)
        sys.stdout.write("Loading BERT tokenizer...\n")
        tokenizer = tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
    elif model_chose == "longformer":
        sys.stdout.write("Loading Longformer base model...\n")
        max_len = 4096
        model = LongformerForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",
            num_labels = NUM_LABELS,
            output_attentions = False,
            output_hidden_states = False)
        sys.stdout.write("Loading Longformer tokenizer...\n")
        tokenizer = LongformerTokenizer.from_pretrained(
            'allenai/longformer-base-4096', do_lower_case=True)
    else:
        raise ValueError("Model type is not supported")
    model.to(DEVICE)
    #############################
    # training transcripts
    trans = df_to_train["text"].values.tolist()
    train_labels = torch.tensor(
        df_to_train["label"].values.tolist())
    input_ids, attn_masks = format_input(
        trans, tokenizer, max_len)
    train_loader = set_dataloader(
        input_ids, attn_masks, train_labels)
    ############################
    # validation transcripts
    val_trans = df_to_val["text"].values.tolist()
    val_labels = torch.tensor(
        df_to_val["label"].values.tolist())
    val_input_ids, val_attn_masks = format_input(
        val_trans, tokenizer, max_len)
    val_loader = set_dataloader(
        val_input_ids, val_attn_masks, val_labels)
    ft_path = os.path.join("../fine_tuned/", ft_model_name)
    if not os.path.exists(ft_path):
        fine_tune_process(train_loader, model, ft_model_name)
    sys.stdout.write("Loading fine-tuned model...\n")
    model_ft = AutoModelForSequenceClassification.from_pretrained(ft_path)
    val_process(val_loader, model_ft)


def cross_validate_process():
    """
    the final process of fine-tuning and validating process
    """
    start_time = datetime.datetime.now()
    args = parse_args()
    # to change GPU usage
    # run the following command on terminal
    # export CUDA_VISIBLE_DEVICES=2
    # set logging file
    FILE_NAME = "{}_fine_tune_batch_{}_epoch_{}.log".format(
        args.model, BATCH_SIZE, EPOCHS)
    log = open(FILE_NAME, "w")
    sys.stdout = log
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        filemode="a", level=logging.INFO,
        filename=FILE_NAME)
    # load dataset
    db_df = pd.read_csv("data/db.tsv", sep="\t")
    db_df["label"] = db_df["label"].astype(int)
    ccc_df = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    adr_full = pd.read_csv("data/adress_full.tsv", sep="\t")
    adr_full["label"] = adr_full["label"].astype(int)
    # merge multiple transcripts belong to single participants
    db_df = db_df.groupby(["file", "label"])["text"].apply(
        lambda x: " ".join(x)).reset_index()
    ccc_df = ccc_df.groupby(["file", "label"])["text"].apply(
        lambda x: " ".join(x)).reset_index()
    sys.stdout.write("################################\n")
    sys.stdout.write("################################\n")
    sys.stdout.write("Fine tuning on ADR full, Validating on DB\n")
    fine_tune_driver(
        adr_full, db_df, args.model, "adr_" + args.model)
    sys.stdout.write("################################\n")
    sys.stdout.write("################################\n")
    sys.stdout.write("Fine tuning on ADR full, Validating on CCC\n")
    fine_tune_driver(
        adr_full, ccc_df, args.model, "adr_" + args.model)
    sys.stdout.write("################################\n")
    sys.stdout.write("Fine tuning on DB, Validating on ADReSS full\n")
    fine_tune_driver(
        db_df, adr_full, args.model, "db_" + args.model)
    sys.stdout.write("################################\n")
    sys.stdout.write("################################\n")
    sys.stdout.write("Fine tuning on DB, Validating on CCC\n")
    fine_tune_driver(
        db_df, ccc_df, args.model, "db_" + args.model)
    sys.stdout.write("################################\n")
    sys.stdout.write("################################\n")
    sys.stdout.write("Fine tuning on CCC, Validating on ADReSS full\n")
    fine_tune_driver(
        ccc_df, adr_full, args.model, "ccc_" + args.model)
    sys.stdout.write("################################\n")
    sys.stdout.write("################################\n")
    sys.stdout.write("Fine tuning on CCC, Validating on DB\n")
    fine_tune_driver(
        ccc_df, db_df, args.model, "ccc_" + args.model)
    sys.stdout.write("Total running time: {}\n".format(datetime.datetime.now()-start_time))
    log.close()


if __name__ == '__main__':
    cross_validate_process()
