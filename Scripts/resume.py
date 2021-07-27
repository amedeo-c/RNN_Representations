
from network import Mikolov
import utilities
from trainer import bptt_trainer, validator
import torch
import torchtext.data as data
import torchtext.datasets as datasets

"""
Equivalent to experiment.py but here model parameters are initialized from a previous
saved checkpoint.
"""

TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), init_token='<SOS>', eos_token='<EOS>', lower=True)

tweets = utilities.extract_tweets()
train_tweets = utilities.dataset_from_text(tweets, TEXT)
print("tweets dataset constructed.")
train_ptb, valid_ptb, test_ptb = datasets.PennTreebank.splits(TEXT, root="treebank.data")
print("PTB datasets constructed.")
TEXT.build_vocab(train_ptb, valid_ptb, test_ptb, train_tweets) #9.733 with only PTB, 27.780 in total
print("Vobabulary built.")

model = Mikolov(len(TEXT.vocab))
train_iter = data.BPTTIterator(train_ptb, batch_size=1, bptt_len=64)
valid_iter = data.BPTTIterator(valid_ptb, batch_size=1, bptt_len=64)

valid_losses = [5]
for i in range(1,15): # the first .pt file results being corrupted.
    epoch = i+1
    path = ("saved/treebank_for_" + str(epoch) + "_epochs.pt")
    checkpoint = torch.load(path)
    print("checkpoint #" + str(epoch) + " loaded.")
    valid_loss = checkpoint['loss_on_validation']
    print("Validation loss after epoch #" + str(epoch) + ": " + str(valid_loss))
    valid_losses.append(valid_loss)

last_checkpoint = torch.load("saved/treebank_for_15_epochs.pt") # important to change this according to the last checkpoint
model.load_state_dict(last_checkpoint['model_state_dict'])
epochs = 15 # important to change this according to the last checkpoint
while True:
    print("Training epoch #" + str(epochs+1) + " starts.")
    bptt_trainer(model, train_iter)
    epochs += 1
    print("Epoch #" + str(epochs) + " completed.")
    valid_loss = validator(model, valid_iter)
    print("Averaged loss on validation set: " + str(valid_loss))
    valid_losses.append(valid_loss)
    path = ("saved/treebank_for_" + str(epochs) + "_epochs.pt")
    torch.save({'model_state_dict': model.state_dict(), 'loss_on_validation': valid_loss}, path)
    # if the validation loss increases twice in a row or we reach 20 epochs, we stop training.
    if (len(valid_losses) >= 3) and (valid_losses[epochs-1] > valid_losses[epochs-2]) and (valid_losses[epochs-2] > valid_losses[epochs-3]):
        print("Training completed.")
        break
    if epochs >= 20:
        print("Training completed.")
        break
