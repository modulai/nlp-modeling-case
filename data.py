import os
import collections
import pandas as pd
import numpy as np
import torch
import torchtext
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


def split_data(df, percent_train):
    """
    Split data in training and validation sets.
    TODO: adding a test set is recommended

    Input:
        - df (pandas.DataFrame): filename to csv file to load 
        - percent_train (float): percent data used for training

    Output:
        - df_train (array): tweets for training
        - df_train_y (array): labels for training
        - df_eval_x (array): tweets for evaluation
        - df_eval_y (array): labesl for evaluation
    """
 
    # convert the datetime column to pandas datatime format
    df['datetime_pd'] = pd.to_datetime(df.datetime)
    # sort tweets based on time, we prefer not to have leakage in time
    df_sorted_time = df.sort_values('datetime_pd')
    # count number of rows in dataset
    no_rows = df_sorted_time.shape[0]
    # get training data
    df_train = df_sorted_time[:int(no_rows*percent_train)]
    # get evaluation data
    df_eval = df_sorted_time[int(no_rows*percent_train):]
    # get training tweets
    df_train = df_train
    # get training labels
    #df_train_y = df_train.is_positive
    # get evaluation tweets 
    df_eval = df_eval
    # get evaluation labels
    #df_eval_y = df_eval.is_positive

    return df_train, df_eval


def tokenize_tweet(tweet, tokenizer, max_length):
    """
    Tokenize a single tweeet

    Input:
        - tweet (str): tweet to tokenize
        - tokenizer (torchtext.tokenizer?): tokenizer
        - max_length (int): maximum length of tokens

    Output:
        - tokens (list(str)): list of tokens
    """

    tokens = tokenizer(tweet)[:max_length]
    return tokens


def tokens_to_integers(tokens, vocab):
    """
    Converts tokens to integers

    Input:
        - tokens (list(str)): tokens to convert 
        - vocab (torchtext.vocab): vocabulary for tokens

    Output:
        - ids (list(int)): indices for tokens in vocabulary
    """
    
    ids = vocab.lookup_indices(tokens)
    return ids


class CustomDataset(Dataset):
    """
    Create a custom Dataset to be used with a DataLoader
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # TODO: it is not efficient to go to numpy array and then to torch tensor
        X = torch.from_numpy(np.array(self.X[idx]))
        y = torch.from_numpy(np.array(self.y[idx]))
        sample = {"X": X, "y": y}
        return sample


def get_collate_fn(pad_index):
    """
    Function to merge a list of samples to form a mini-batch of padded Tensors.

    Input:
        pad_index (int): index of "padding" token in vocabulary

    Output:
        collate_fn: function to collate samples
    """
    
    def collate_fn(batch):
        batch_ids = [i["X"] for i in batch]
        # pad sequences to make sure all samples are of the same length
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["y"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"X": batch_ids, "y": batch_label}
        return batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    """
    Create a DataLoader to get an iterable over the dataset

    Input: 
        - dataset (DataSet): dataset to use with dataloader
        - batch_size (int): batch size for training
        - pad_index (int): index of "padding" token in vocabulary
        - shuffle (bool): whether the data should be shuffled or not
    """
    
    # collate function to create mini-batches
    collate_fn = get_collate_fn(pad_index)

    # Use a DataLoader to facilitate data loading
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        drop_last=True
    )

    return data_loader


def load_and_process_data(filename='twitter_dataset_small_w_bart_preds', percent_train=0.9, batch_size=512):
    """
    Loads and process the data to a format that can be used for training.

    Input:
        - filename (str): filename of file to load
        - percent__train (float): percent of data to use for training set and rest is used for evaluation
        - batch_size (int): batch size for training

    Output:
        - train_data_loader: data loader for training
        - eval_data_loader: data loader for evaluation
        - vocab_size: vocabulary size
        - pad_index: index of "padding" token in vocabulary

    """

    # load the data into a DataFrame
    # TODO: allow different fileformats
    df = pd.read_csv(os.path.join("data", filename))
    # adding an assumption that the tweets are in English language
    # TODO: experiment with different tokenizers
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    # set a max length of tokens
    # TODO: refine this number as tweets are limitied to 280 characters
    max_length = 256
    # tokenize tweets and add as column to DataFrame
    df['tokens'] = df.apply(lambda x: tokenize_tweet(x['message'], tokenizer, max_length), axis=1)
    # split data into training and evaluation sets
    # TODO: good to add a test set as well
    df_train, df_eval = split_data(df, percent_train)

    # set the minimum count of a token to be part of the vocabulary
    min_freq = 5
    # define two special tokens, for "unknown" tokens and "padding"
    special_tokens = ["<unk>", "<pad>"]

    # use torchtext function to build a vocabulary
    vocab = torchtext.vocab.build_vocab_from_iterator(
        df_train['tokens'],
        min_freq=min_freq,
        specials=special_tokens,
    )

    # get index of the "unknown" token in the vocabulary
    unk_index = vocab["<unk>"]
    # get index of the "padding" token in the vocabulary
    pad_index = vocab["<pad>"]
    # if the token cannot be found in the vocabulary use the "unknown" token
    vocab.set_default_index(unk_index)

    # convert tokens to integers, using the indices of the tokens in the vocabulary
    df_train['tokens_num'] = df_train.apply(lambda x: tokens_to_integers(x['tokens'], vocab), axis=1)
    df_eval['tokens_num'] = df_eval.apply(lambda x: tokens_to_integers(x['tokens'], vocab), axis=1)

    # Create a custom Torch DataSet for training to facilitate training
    train_dataset = CustomDataset(df_train['tokens_num'].values, df_train['is_positive'].values)
    # Create a custom Torch DataSet for evaluation to facilitate evaluation
    eval_dataset = CustomDataset(df_eval['tokens_num'].values, df_eval['is_positive'].values)

    # create a data loader for training data
    train_data_loader = get_data_loader(train_dataset, batch_size, pad_index, shuffle=True)
    # create a data loader for evaluation data
    eval_data_loader = get_data_loader(eval_dataset, batch_size, pad_index, shuffle=False)

    # calculate the size of the vocabulary
    vocab_size = len(vocab)

    return train_data_loader, eval_data_loader, vocab_size, pad_index
    
