
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mikolov(nn.Module):
    """
    The model takes indexes (in the vocabulary) as inputs. The  shape is (sequence_length, batch_size).
    The indexes are transformed to feature vectors (embeddings) by the embedding layer, and in turn these
    are sequentially fed in a recurrent layer. Then, the model outputs log_probabilities
    with shape (sequence_length, batch_size, vocabulary_size).
    """
    def __init__(self, vocab_size, embedding_size = 128, hidden_size=128):
        super(Mikolov, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size)
        self.gen = Generator(vocab_size, hidden_size)
        self.vocabulary_size = vocab_size

    def forward(self, x):
        embeddings = self.embed(x)
        output, hidden = self.rnn(embeddings)
        return self.gen(output)

    def get_embeddings(self):
        """
        In this case, word representations are learned in the embedding layer, specifically
        in the rows of its (vocab_size, embedding_size) weight matrix.
        """
        return self.embed.weight

class Original_Mikolov(nn.Module):
    """
    following Mikolov et al., 2011.
    Differently compared to the above, this model takes vocab_size-dimensional one hot
    encoding vectors as inputs. The embedding layer is removed, so that word representations are learned
    in the rnn layer (which incorporates a projection from vocab_size to embedding_size).
    """
    def __init__(self, vocab_size, hidden_size=128):
        super(Original_Mikolov, self).__init__()
        self.rnn = nn.RNN(vocab_size, hidden_size, bias = False)
        self.gen = Generator(vocab_size, hidden_size)
        self.vocabulary_size = vocab_size

    def forward(self, x):
        output, hidden = self.rnn(x.float())
        return self.gen(output)

    def get_embeddings(self):
        """
        """
        return list(self.parameters())[0]


class Generator(nn.Module):
    """
    this module takes as input a hidden_size-dimensional vector (in this case the current hidden state
    of the network). It projects it on the vocabulary and then applies log_softmax, so that we can
    interpret output as log_probabilities.
    """
    def __init__(self, vocab_size, hidden_size):
        super(Generator, self).__init__()
        self.projection = nn.Linear(hidden_size, vocab_size, bias = False)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=-1)
