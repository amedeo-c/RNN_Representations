
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from statistics import mean
#from torch.utils.tensorboard import SummaryWriter

"""
Utility functions used in the main training algorithm.
"""

def bptt_trainer(model, iterator):
    """
    This performs one whole epoch of training.
    Our algorithm implements TBPTT with k1 = k2 = batch_size*sequence_length = k. We process k contiguous
    tokens and then perform an optimization step based on the (averaged) error on all of them.
    """
    #writer = SummaryWriter("runs/experiment_1")
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)

    processed = 0
    model.train()
    for batch in iterator:

        optimizer.zero_grad()
        input = batch.text
        output = model(input)
        loss = loss_function(output.view(-1, model.vocabulary_size), batch.target.view(-1))
        loss.backward()
        optimizer.step()

        # this is only to check training evolution on console
        processed += torch.numel(input)
        if processed % (torch.numel(input)*100) == 0:
            print("Tokens processed: " + str(processed))

    model.eval()

def validator(model, iterator):
    """
    This only calculates (aggregated) loss on the validation set.
    """
    loss_function = nn.NLLLoss()
    losses = []
    for batch in iterator:
        input = batch.text
        output = model(input)
        loss = loss_function(output.view(-1, model.vocabulary_size), batch.target.view(-1))
        losses.append(loss.item())
    return mean(losses)

def original_bptt_trainer(model, iterator):
    """
    """
    #writer = SummaryWriter("runs/experiment_1")
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)
    processed = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        input = F.one_hot(batch.text.long(), model.vocabulary_size)
        output = model(input)
        loss = loss_function(output.view(-1, model.vocabulary_size), batch.target.view(-1))
        loss.backward()
        optimizer.step()

        # this is only to check training evolution on console
        processed += torch.numel(batch.text)
        if processed % (torch.numel(batch.text)*100) == 0:
            print("Tokens processed: " + str(processed))

    model.eval()

def original_validator(model, iterator):
    loss_function = nn.NLLLoss()
    losses = []
    for batch in iterator:
        input = F.one_hot(batch.text.long(), model.vocabulary_size)
        output = model(input)
        loss = loss_function(output.view(-1, model.vocabulary_size), batch.target.view(-1))
        losses.append(loss.item())
    return mean(losses)

"""
run the tensorboard process at port 6006 using the following command:
tensorboard --logdir=runs
"""
