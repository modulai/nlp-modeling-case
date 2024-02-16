### Solution

This repository provides a solution to train a simple neural version of "bag of words" for sentiment analysis.


#### Requirements
There are a number of dependencies on python libraries.  
Install pytorch according to best practices: https://pytorch.org/  
For the rest of the requirements run:
```
pip install -r requirements.txt
```


#### File structure
The repository is structured in:  
- data.py: This is where the logic for cleaning and tokenizing the data, splitting the data in training and validation sets
- model.py: This is where the logic for defining the model and the parameters around it is placed.
- train.py: This is where the logic for training and evaluating the model is placed. 
- train_model.py: The main entrypoint for training a new model
- evaluate_model.py: The main entrypoint for evaluating a trained model
- data_exploration.ipynb: Notebook with very basic exploration of the data
- results.ipynb: Notebook to plot accuracy and loss curves of an experiment
- bart_example.ipynb: Example given on how to do predictions with a pretrained BART model

#### Train model

To train the model with default parameters:  
```
python train_model.py
```

This results in an experiment folder being created, storing input parameters, network weights, and accuracies and losses.


#### Plot results

Use results.ipynb to look at the results from the experiments.


#### Evaluate model

To evaluate a trained model with default paramters:

```
python evaluate_model.py --exp_id 
```

