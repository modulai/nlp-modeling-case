import argparse

import data, train, model


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # general parameters
    parser.add_argument('--seed', default='12345', type=int,
                        help=f'seed to use for replicability')
    parser.add_argument('--exp_id', default=-1, type=int,
                        help=f'experiment id to use')

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


def main():
    """
    Main function for model evaluation.
    """

    # Load parameters
    seed = FLAGS.seed
    filename = FLAGS.filename
    model_name = FLAGS.model_name
    percent_train = FLAGS.percent_train
    embedding_dim = FLAGS.embedding_dim
    exp_id = FLAGS.exp_id

    if exp_id == -1:
        raise ValueError("You have to give an experiment id to use.")

    # Load and process data
    _, eval_data_loader, vocab_size, pad_index = data.load_and_process_data(filename, percent_train)
    
    # Setup the model
    model_fn, criterion, optimizer, device = model.setup_model(model_name, vocab_size, embedding_dim, pad_index, seed, exp_id)

    # Evaluate model
    loss, accuracy = train.evaluate_epoch(eval_data_loader, model_fn, criterion, optimizer, device)
    print("Evaluation loss: {}, Evaluation accuracy: {}".format(loss, accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()

    main()
