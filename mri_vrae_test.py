"""
Time-series clustering in a py script.

Easier for me to run this and debug it, jupyter from hpc is messy.

Leave jupyter notebooks for generating 
"""

from vrae.vrae import VRAE
from vrae.utils import open_MRI_data
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

### Parameters
hidden_size = 20
hidden_layer_depth = 1
latent_length = 5
learning_rate = 0.0005
n_epochs = 8000
dropout_rate = 0.0 # We have variational dropout in our implementation
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every = 1000
clip = True # options: True, False # Gradient Clipping
max_grad_norm = 5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'GRU' # options: LSTM, GRU

dload = './mri_model_dir' #download directory

# LOAD DATA
csv_path = "data/tadpole_mrionly.csv"
X_train, X_val = open_MRI_data(csv_path, train_set=0.8, n_followups=5, normalize=True)

sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]

#We define batch size as the full length
# To change this we would need to change structurally everything. Afternoon.
# batch_size = X_train.shape[0] #In our original paper, there is no batch size, all the optimization is done together
batch_size = len(X_train)

#Convert to torch
#Why use TENSORDATASET? In theory, no need to use it
train_dataset = TensorDataset(torch.from_numpy(X_train))
test_dataset = TensorDataset(torch.from_numpy(X_val))

#We would like the batch size 
# initiate VRAE and fit
vrae = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer, 
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)

# Fit the model
vrae.fit(train_dataset)

print(vrae.encoder)
print(vrae.lmbd)
print(vrae.decoder)

# Evaluate results:
#     * plot loss
#     * compute a metric for the testing dataset
loss_curve = vrae.training_loss
print(np.array(loss_curve).shape)
#Plot it
plt.plot(range(len(loss_curve)), loss_curve, '-b', label='loss')

plt.xlabel("iteration")
plt.ylabel("total loss")

plt.legend(loc='upper left')
plt.title("Loss function")

plt.savefig(dload + 'loss.png')
plt.close()
# Transform the test dataset
X_hat_train = vrae.reconstruct(train_dataset)
X_hat = vrae.reconstruct(test_dataset)

# Need to reshape
X_hat = np.swapaxes(X_hat,0,1)
X_hat_train = np.swapaxes(X_hat_train,0,1)

print(X_hat.shape)
print(X_val.shape)

#Compute mean absolute error over all sequences
mse_test = np.mean([mean_absolute_error(xval, xhat) for (xval, xhat) in zip(X_train, X_hat_train)])
print('MSE over the train set: ' + str(mse_test))

#Compute mean absolute error over all sequences
mse_test = np.mean([mean_absolute_error(xval, xhat) for (xval, xhat) in zip(X_val, X_hat)])
print('MSE over the test set: ' + str(mse_test))


# Visualize the difference between X_test and X_hat for specific features and 
subject = 12
feature = 0

x_hat_curve = X_hat[subject, :, feature]
x_val_curve = X_val[subject, :, feature]

# Plot the two lines
plt.plot(range(len(x_hat_curve)), x_hat_curve, '-b', label='X (predicted)')
plt.plot(range(len(x_val_curve)), x_val_curve, '-r', label='X (original)')

plt.xlabel("time-point")
plt.ylabel("value")

plt.legend(loc='upper left')
plt.title("Predicted vs real")

plt.savefig(dload + 'predicted.png')
plt.close()

# Visualize the difference between X_test and X_hat for a single subject, using the mean of the predictions and of all the features
subject = 13

x_hat_curve = np.mean(X_hat[subject, :, :], axis=1)
x_val_curve = np.mean(X_val[subject, :, :], axis=1)

# Plot the two lines
plt.plot(range(len(x_hat_curve)), x_hat_curve, '-b', label='X (predicted)')
plt.plot(range(len(x_val_curve)), x_val_curve, '-r', label='X (original)')

plt.xlabel("time-point")
plt.ylabel("value")

plt.legend(loc='upper left')
plt.title("Predicted vs real")

plt.savefig(dload + 'predicted_subj.png')
plt.close()