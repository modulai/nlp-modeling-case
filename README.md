![Twitter](twitter.jpg)

# NLP Modeling Case

## Description

The task is to use a version of the Twitter sentiment dataset (you can download the dataset using this [link](https://drive.google.com/file/d/13mAaFqCrscUYkoITf4rZ6qG9ptAlIJVb/view?usp=sharing)) and create a comprehensive preprocessing, training, and validation pipeline to predict the target variable `is_positive`. Additionally, you are required to compare your results with those obtained using Zero-shot prediction with a pre-trained [BART model](https://huggingface.co/transformers/model_doc/bart.html).

The repository contains a small version of the dataset (20K examples) with an additional column `bart_is_positive`, which contains the Zero-shot prediction of the BART model using the query `This example is positive`. An example of how this is done is included in the notebook [bart_example.ipynb](./bart_example.ipynb). If you do not have access to a GPU (as BART inference can be slow using CPU), you are free to base your training and analysis on this smaller version of the dataset.

### Summary

The coding task consists of two parts. First, you need to predict the target variable `is_positive` using your chosen method. Second, you need to perform inference on the same texts using the output of the BART model and compare the results of the two models. We provide predictions for this step in the smaller version of the dataset to assist you.

## Tasks

1. Read the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) to gain a high-level understanding of the architecture, dataset creation, and data encoding. Be prepared to discuss it briefly during the interview.

2. Load the dataset (small or full version).

3. Perform preprocessing and cleaning steps of your choice.

4. Train a model (Choose a method/architecture you find suitable).

5. Validate the model's performance in terms of predictive accuracy and generalizability. You are free to choose relevant metrics for this task.

6. Make additional predictions using the BART model (or use the small dataset with the precomputed predictions) and validate the results. Compare these results to the ones obtained from the model trained in the previous step.

7. Present your code, descriptive analysis, and model performance, for example, in a Jupyter notebook.

## Final note

If you have limited time or encounter difficulties, do as much as you can and be prepared to explain any omitted steps. This is an open-ended case, and you are encouraged to solve it in a way that you find suitable. Some statements may be vague, such as "compare" the models, and in such cases, you are free to make your own interpretations and assumptions.
