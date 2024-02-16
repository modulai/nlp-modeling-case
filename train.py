import collections
import tqdm 
import torch
import numpy as np
import pandas as pd


def get_accuracy(prediction, label):
    acc = (prediction.round() == label).float().mean()
    acc = float(acc)

    return acc


def train_epoch(data_loader, model, criterion, optimizer, device):
    """
    Main training loop for an epoch 
    """
    
    model.train()
    epoch_losses = []
    epoch_accs = []

    # start training loop for this epoch
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["X"].to(device)
        label = torch.unsqueeze(batch["y"].to(device),1).to(torch.float32)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy)
    return np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate_epoch(data_loader, model, criterion, optimizer, device):
    """
    Main evaluation loop for an epoch 
    """
    
    model.train()
    epoch_losses = []
    epoch_accs = []

    # start training loop for this epoch
    for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
        ids = batch["X"].to(device)
        label = torch.unsqueeze(batch["y"].to(device),1).to(torch.float32)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy)
    return np.mean(epoch_losses), np.mean(epoch_accs)


def train_and_evaluate_model(exp_id, epochs, model, train_data_loader, eval_data_loader, criterion, optimizer, device):

    best_valid_loss = float("inf")

    metrics = collections.defaultdict(list)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            train_data_loader, model, criterion, optimizer, device
        )
        valid_loss, valid_acc = evaluate_epoch(eval_data_loader, model, criterion, optimizer, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "experiments/{}/weights.pt".format(exp_id))
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
    pd.DataFrame(metrics).to_csv("experiments/{}/results.csv".format(exp_id), index=False)
    