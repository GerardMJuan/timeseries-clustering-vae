"""
New file to implement the changes onto vrae and the other classes.
We will start anew, copying over and making necessary changes. This way, we can 
leave vrae.py untouched and working.
"""

import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence, pad_packed_sequence
import os

## Decidint on device on device.
DEVICE_ID = 0
DEVICE = torch.device('cuda:' + str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(DEVICE_ID)


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    #Sort the data
    batch = sorted(batch, key=len, reverse=True)

    lengths = torch.tensor([ t.shape[0] for t in batch ])
    ## padd
    batch = [ torch.FloatTensor(t) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    return batch.to(device=DEVICE_ID), lengths.to(device=DEVICE_ID)

def pack_sequence(X, pad=True, lengths=None):
    """
    Function to pack a batch of different sequence length.
    """
    if pad:
        # Sort the data by sequence length
        X = sorted(X, key=len, reverse=True)
        # Pad the sequence
        #and get length of the sequence
        lengths = [len(x) for x in X]
        X = pad_sequence(X) 
    # Pack the sequence
    X = pack_padded_sequence(X, lengths)
    # Update lengths
    return (X, lengths)

class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.block = block

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x, lengths):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        x_packed = pack_padded_sequence(x, lengths)


        if self.block == 'LSTM':
            _, (h_end, c_end) = self.model(x_packed)
        elif self.block == 'GRU':
            _, h_end = self.model(x_packed)
        else:
            raise NotImplementedError

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            # Reparametrization trick
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean


class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence. Needed to implement the dummy zeros for the output
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, lengths):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)
        #How to create the inputs of the decoder? With a variable sequence length?
        # We have the length and the batch size
        max_length = torch.max(lengths) # get the max length
        padded_inputs = torch.zeros(max_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.decoder_inputs = pack_padded_sequence(padded_inputs, lengths)
        
        # Only needed when using LSTM
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError
        
        # Unppack
        decoder_unpacked, lens_unpacked = pad_packed_sequence(decoder_output)

        #This implements a linear layer between the output of the lstm and the final shape
        # here could introduce a breakpoint to check the sizes of everything
        out = self.hidden_to_output(decoder_unpacked)
        return out


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

"""
Main changes w.r.t. previous version:
* It uses paddedsequences. This should be done in a separate function, after feeding the data.
Check this when debugging:
https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence
"""

class VRAE(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    """
    def __init__(self, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss',
                 cuda=False, print_every=100, clip=True, max_grad_norm=5, dload='.'):

        super(VRAE, self).__init__()


        self.dtype = torch.FloatTensor
        self.use_cuda = cuda

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False


        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor

        #Initialize individual components
        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)

        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload

        # lengths of the current packedsequence, to use in the decoder
        self.lengths = None

        #self.training_loss = None

        self.init_loss()

        if self.use_cuda:
            self.cuda()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output = self.encoder(x, self.lengths)
        latent = self.lmbd(cell_output)
        #Pass as input the shape of the original data, which will be equivalent to the batch size
        x_decoded = self.decoder(latent, self.lengths)
        return x_decoded, latent

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss

    def compute_loss(self, X):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration

        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        # x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, _ = self(X)
        loss, recon_loss, kl_loss = self._rec(x_decoded, X, self.loss_fn)
        losses = {'total': loss,
                  'kl': kl_loss,
                  'll': recon_loss}

        if self.training:
            self.save_loss(losses)
            return loss
        else:
            return losses

    def save_loss(self, losses):
        for key in self.loss.keys():
            self.loss[key].append(float(losses[key].detach().item()))

    def init_loss(self):
        empty_loss = {
            'total': [],
            'kl': [],
            'll': []
        }
        self.loss = empty_loss


    def _train(self, X):
        """
        For each epoch run this function batch_size * num_of_batches number of times

        :param train_loader: input data
        :return: returns the average loss
        """
        self.train() # Inherited method which sets self.training = True
        # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
        # It is possible that this is no longer needed?
        # X = X.permute(1,0,2)

        self.optimizer.zero_grad()
        loss = self.compute_loss(X)
        loss.backward()

        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)

        self.optimizer.step()

        return loss

    def fit(self, dataset, save = False):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`
        : dataset: a TensorDataset
        :param data: list of sequences of variable length, list[Tensor], or a PackedSequence object
        https://discuss.pytorch.org/t/packedsequence-with-dataloader/1495
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """
        
        train_loader = DataLoader(dataset = dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last = True,
                                  collate_fn = collate_fn_padd)

        for i in range(self.n_epochs):
            t = 0
            #For each epoch ,run the train loader
            for t, (X, lengths) in enumerate(train_loader):
                #Get 
                self.lengths = lengths
                # Convert to packetsequence
                # data, _ = pack_sequence(X, False, self.lengths)
                # Run
                loss = self._train(X)
                loss = loss.detach().cpu()
            #Check if nan
            if np.isnan(loss):
                print('Loss is nan!')
                break

            if i % self.print_every == 0:
                print('Epoch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (i, self.loss['total'][i],
                                                                                    self.loss['ll'][i], self.loss['kl'][i]))
                print('Average loss: {:.4f}'.format(loss))
            
        self.is_fitted = True

        if save:
            self.save('model.pth')

        self.eval()  # Inherited method which sets self.training = False



    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function

        :param x: input batch tensor
        :return: intermediate latent vector
        """
        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )
        ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function

        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, data, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit

        :param data: input data who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        # Check if the data is a padded sequence. If not, pad it
        if not isinstance(data, PackedSequence):
            data, self.lengths = pack_sequence(data)


        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []
                # x = x[0]
                # x = x.permute(1, 0, 2)
                x_decoded = self._batch_reconstruct(data)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')


    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        # Check if the data is a padded sequence. If not, pad it
        if not isinstance(data, PackedSequence):
            data, self.lengths = pack_sequence(data)


        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                #x = x[0]
                #x = x.permute(1, 0, 2)
                z_run = self._batch_transform(data)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above

        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later

        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned

        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))




class VRAE_seq(VRAE):

    def __init__(self, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss',
                 cuda=False, print_every=100, clip=True, max_grad_norm=5, dload='.'):

        super(VRAE, self).__init__()


