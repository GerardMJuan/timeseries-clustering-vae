from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_clustering(z_run, labels, engine ='plotly', download = False, folder_name ='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """
    """
    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)
    """
    def plot_clustering_matplotlib(z_run, labels, download, folder_name):

        labels = labels[:z_run.shape[0]] # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('PCA on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/pca.png")
        else:
            plt.show()

        plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('tSNE on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/tsne.png")
        else:
            plt.show()

    #if (download == False) & (engine == 'plotly'):
    #    plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)


def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


def pandas_to_data_timeseries(df, feat, n_timesteps = 5, id_col = 'PTID', time='VISCODE'):
    """
    Quick function that converts a pandas dataframe with the features
    indicated by a vector "feat" (with the name of the features columns) and "id_col"
    indicating the column of the subject ids, and column time to order them by time.

    We assume that the data is already preprocessed so that, for each PTID, there are n_timesteps
    rows.
    """
    # Order the dataframe
    df = df.sort_values(by=[id_col, time], ascending=False)

    #Nuumber of samples
    sample_list = np.unique(df[id_col])

    # Create base numpy structure
    X = np.zeros((len(sample_list), n_timesteps, len(feat)))
    # Iterate over each subject and fill it
    df_feats = df.loc[:, feat]
    i = 0
    for ptid in sample_list:
        i_list = df.index[df['PTID'] == ptid]
        feats = df_feats.iloc[i_list, :].values
        X[i, :, :] = feats
        i += 1

    # Return numpy dataframe
    return X

def pandas_to_data_timeseries_var(df, feat, id_col = 'PTID', time='VISCODE'):
    """
    Quick function that converts a pandas dataframe with the features
    indicated by a vector "feat" (with the name of the features columns) and "id_col"
    indicating the column of the subject ids, and column time to order them by time.

    The number of rows is variable, so we are creating a list of numpy arrays
    """
    # Order the dataframe
    df = df.sort_values(by=[id_col, time], ascending=False)

    #Nuumber of samples
    sample_list = np.unique(df[id_col])

    # Create base list
    X = []
    # Iterate over each subject and fill it
    df_feats = df.loc[:, feat]
    i = 0
    for ptid in sample_list:
        i_list = df.index[df['PTID'] == ptid]
        feats = df_feats.iloc[i_list, :].values
        X.append(feats)
        i += 1

    # Return numpy dataframe
    return X

def open_MRI_data(csv_path, train_set = 0.8, n_followups=5, normalize=True):
    """
    open MRI data from the specified directory
    We only return subjects with n_followups. If less, not included. If more, truncated.
    Divide between test and train.
    Return with the correct format (Nsamples, timesteps, nfeatures)
    (normalize parameter not used)

    NOTE: NEW VERSION SHOULD SORT THE DATA SEQUENCES FROM LONG TO SHORT, AND OUTPUTS ACCORDINGLY
    """

    data_df = pd.read_csv(csv_path)

    mri_col = data_df.columns.str.contains("SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16")
    mri_col = data_df.columns[mri_col].values

    data_df = data_df.dropna(axis=0, subset=mri_col)

    # Select only the subjects with nfollowups
    # Code to only select 5 first appearances of each PTID
    ptid_list = np.unique(data_df["PTID"])

    idx_to_drop = []
    for ptid in ptid_list:
        i_list = data_df.index[data_df['PTID'] == ptid].tolist()
        if len(i_list) < 5:
            idx_to_drop = idx_to_drop + i_list
        elif len(i_list) > 5:
            idx_to_drop = idx_to_drop + i_list[5:]

    data_final = data_df.drop(idx_to_drop)

    print(data_final.shape)

    # Normalize only features
    data_final.loc[:,mri_col] = data_final.loc[:,mri_col].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

    # Divide between test and train
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=1.0-train_set)
    train_dataset, test_dataset = next(gss.split(X=data_final, y=data_final.DX_bl.values, groups=data_final.PTID.values))

    df_train = data_final.iloc[train_dataset]
    df_test =  data_final.iloc[test_dataset]

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Return the features in the correct shape (Nsamples, timesteps, nfeatures)
    X_train = pandas_to_data_timeseries(df_train, mri_col)
    X_test = pandas_to_data_timeseries(df_test, mri_col)

    return X_train, X_test


def open_MRI_data_var(csv_path, train_set = 0.8, normalize=True):
    """
    Function to return a variable number of followups from a dataset

    Returns:
    X_test: list composed of tensors of variable length
    X_train: list composed of tensors of variable length
    """
    data_df = pd.read_csv(csv_path)

    mri_col = data_df.columns.str.contains("SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16")
    mri_col = data_df.columns[mri_col].values

    data_df = data_df.dropna(axis=0, subset=mri_col)

    # Select only the subjects with nfollowups
    # Code to only select 5 first appearances of each PTID
    ptid_list = np.unique(data_df["PTID"])

    idx_to_drop = []
    data_final = data_df.drop(idx_to_drop)

    # Divide between test and train
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=1.0-train_set)
    train_dataset, test_dataset = next(gss.split(X=data_final, y=data_final.DX_bl.values, groups=data_final.PTID.values))

    df_train = data_final.iloc[train_dataset]
    df_test =  data_final.iloc[test_dataset]

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Return the features in the correct shape list of Tensors (timesteps, nfeatures)
    X_train = pandas_to_data_timeseries_var(df_train, mri_col)
    X_test = pandas_to_data_timeseries_var(df_test, mri_col)

    return X_train, X_test

