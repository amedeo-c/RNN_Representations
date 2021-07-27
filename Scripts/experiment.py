
from network import Mikolov
import utilities
from trainer import bptt_trainer, validator
import torch
import torchtext.data as data
import torchtext.datasets as datasets

"""
This includes the whole experiment: data collection, model creation and model training.
"""

# torchtext field, which defines our type of data and how to process it
TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), init_token='<SOS>', eos_token='<EOS>', lower=True)

# prepare datasets and resulting vocabulary
tweets = utilities.extract_tweets()
train_tweets = utilities.dataset_from_text(tweets[0:int(0.8*len(tweets))], TEXT)
valid_tweets = utilities.dataset_from_text(tweets[int(0.8*len(tweets))+1:], TEXT)
print("Twitter datasets constructed.")

train_ptb, valid_ptb, test_ptb = datasets.PennTreebank.splits(TEXT, root="treebank.data")
print("PTB datasets constructed.")

TEXT.build_vocab(train_ptb, valid_ptb, test_ptb, train_tweets, valid_tweets) #9.733 with only PTB, 27.780 in total
print("Vobabulary built.")

# create model
model = Mikolov(len(TEXT.vocab))
#last_checkpoint = torch.load("saved/treebank_for_20_epochs.pt")
#model.load_state_dict(last_checkpoint['model_state_dict'])

# create iterators for training
# here we train on twitter dataset but training on PTB is equivalent
train_iter = data.BPTTIterator(train_tweets, batch_size=1, bptt_len=64)
valid_iter = data.BPTTIterator(valid_tweets, batch_size=1, bptt_len=64)

epochs = 0
valid_losses = [] # validation
while(True):

    print("Training epoch #" + str(epochs+1) + " starts.")
    bptt_trainer(model, train_iter)
    epochs += 1
    print("Epoch #" + str(epochs) + " completed.")

    valid_loss = validator(model, valid_iter)
    print("Averaged loss on validation set: " + str(valid_loss))
    valid_losses.append(valid_loss)

    # we save a copy of the parameters at each training epoch
    path = ("saved/twitter_for_" + str(epochs) + "_epochs.pt")
    torch.save({'model_state_dict': model.state_dict()}, path)

    # if the validation loss increases twice in a row or we reach 20 epochs, we stop training.
    if (len(valid_losses) >= 3) and (valid_losses[epochs-1] > valid_losses[epochs-2]) and (valid_losses[epochs-2] > valid_losses[epochs-3]):
        print("Training completed.")
        break
    if epochs >= 20:
        print("Training completed.")
        break
