import argparse
import os
import json

import data, train, model


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # general parameters
    parser.add_argument('--seed', default='12345', type=int,
                    help=f'seed to use for replicability')

    # data parameters
    parser.add_argument('--filename', default='twitter_dataset_small_w_bart_preds.csv', type=str,
                      help=f'name of data file twitter_dataset_small_w_bart_preds.csv|twitter_dataset_full.csv')
    parser.add_argument('--percent_train', default=0.9, type=float,
                      help=f'percent of data to use for training') 

    # model parameters                 
    parser.add_argument('--model_name', default='mlp', type=str,
                      help=f'name of model to use')
    parser.add_argument('--embedding_dim', default=300, type=int,
                      help=f'vocabulary size')
    parser.add_argument('--batch_size', default=512, type=int,
                      help=f'batch size in training')
    parser.add_argument('--epochs', default=10, type=int,
                      help=f'number of epochs to train')
    


def setup_experiment(params):
    """
    Setup a new experiment

    Input:
        params (dict): input arguments to application

    Output:
        id (int): experiment id that maps to folder structure
    """
    #create experiments folder
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
        # create a folder for the first experiment
        id = str(1)
    else:
        dirs = os.listdir('experiments')
        dirs_sorted = sorted([int(d) for d in dirs])
        id = str(dirs_sorted[-1] + 1)

    new_dir = 'experiments/{}'.format(id)
    os.makedirs(new_dir)

    #save parameters in experiment directory
    with open(os.path.join(new_dir,"params.json"), "w") as f:
        json.dump(params, f)

    return id
        


def main():
    """
    Main function for model training.
    """

    # Load parameters
    # Convert namespace to dict
    params = vars(FLAGS)
    seed = params['seed']
    filename = params['filename']
    model_name = params['model_name']
    percent_train = params['percent_train']
    embedding_dim = params['embedding_dim']
    epochs = params['epochs']

    # Setup a new experiment
    exp_id = setup_experiment(params)

    # Load and process data
    train_data_loader, eval_data_loader, vocab_size, pad_index = data.load_and_process_data(filename, percent_train)
    
    # Setup the model
    model_fn, criterion, optimizer, device = model.setup_model(model_name, vocab_size, embedding_dim, pad_index, seed)

    # Train and evaluate model
    train.train_and_evaluate_model(exp_id, epochs, model_fn, train_data_loader, eval_data_loader, criterion, optimizer, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()

    main()
