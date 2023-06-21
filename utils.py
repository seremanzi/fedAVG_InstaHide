import numpy as np
import pandas as pd
from conf import conf
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import colorsys
import torch
import torch.nn.functional as F
import os

def label_skew(data,label,K,n_parties,beta,min_require_size = 10):
    """
:param data: Data dataframe
:param label: Label column name
:param K: Number of labels
:param n_parties: Number of parties
:param beta: Dirichlet parameter
:param min_require_size: Minimum data size per point, if below this number, the data will be redistributed to ensure each node has enough data
:return: Split the data to different parties based on the Dirichlet distribution
    """
    y_train = data[label]

    min_size = 0
    partition_all = []
    front = np.array([0])
    N = y_train.shape[0]  # Total number of samples
    # return train_datasets, test_dataset, n_input, number_samples
    split_data = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])

            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            back = np.array([idx_k.shape[0]])
            partition =np.concatenate((front,proportions,back),axis=0)
            partition = np.diff(partition) # Calculate the data distribution for each label based on the splitting points
            partition_all.append(partition)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

            min_size = min([len(idx_j) for idx_j in idx_batch])

    # Split the data based on the indices for each node
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data[j] = data.iloc[idx_batch[j], :]

    return split_data,partition_all


def get_data():

    ### Training data
    train_data = pd.read_csv(conf["train_dataset"])

    train_data,partition_all = label_skew(train_data,conf["label_column"],conf["num_classes"],conf["num_parties"],conf["beta"])
    print("Data distribution for each node:")
    print(partition_all)
    
    train_datasets = {}
    val_datasets = {}
    ## Number of samples for each node
    number_samples = {}

    ## Load datasets, split training data into training and validation sets
    for key in train_data.keys():
        ## Shuffle the data
        train_dataset = shuffle(train_data[key])

        val_dataset = train_dataset[:int(len(train_dataset) * conf["split_ratio"])]
        train_dataset = train_dataset[int(len(train_dataset) * conf["split_ratio"]):]
        train_datasets[key] = train_dataset
        val_datasets[key] = val_dataset

        number_samples[key] = len(train_dataset)

    ## Test set, used to evaluate the model on the Server
    test_dataset = pd.read_csv(conf["test_dataset"])
    test_dataset = test_dataset
    print("Data loading complete!")

    return train_datasets, val_datasets, test_dataset

def init_wandb(run_id=None, config=conf):
    group_name = "fedavg_instahide"

    configuration = config
    agents = ""
    for agent in configuration["models"]:
        agents += agent["model_type"][0]
    job_name = f"M{configuration['N_agents']}_N{configuration['N_rounds']}_S{config['N_subset']}_A{agents}"

    run = wandb.init(
                id = run_id,
                # Set entity to specify your username or team name
                entity="samaml",
                # Set the project where this run will be logged
                project='favg_ih',
                group=group_name,
                # Track hyperparameters and run metadata
                config=configuration,
                resume="allow")

    if os.environ["WANDB_MODE"] != "offline" and not wandb.run.resumed:
        random_number = wandb.run.name.split('-')[-1]
        wandb.run.name = job_name + '-' + random_number
        wandb.run.save()
        resumed = False
    if wandb.run.resumed:
        resumed = True

    return run, job_name, resumed


def load_checkpoint(path, model, restore_path = None):
    loaded = False
    if wandb.run.resumed or restore_path is not None:
        try:
            weights = wandb.restore(path, run_path=restore_path)
            model.load_state_dict(torch.load(weights.name))
            print(f"===== SUCCESFULLY LOADED {path} FROM CHECKPOINT =====")
            loaded = True
        except ValueError:
            print(f"===== CHECKPOINT FOR {path} DOES NOT EXIST =====")            
        except RuntimeError:
            print(f"===== CHECKPOINT FOR {path} IS CORRUPTED =====")
            print("Deleting...")
            files = wandb.run.files()
            for file in files:
                if file.name == path:
                    file.delete()
            print("Deleted. Sorry for the inconveniences")
    return loaded


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def label_to_onehot(target, num_classes=conf["num_classes"]):
    '''Returns one-hot embeddings of scaler labels'''
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def vec_mul_ten(vec, tensor):
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res



def mixup_criterion(pred, ys, lam_batch, klam, num_class=conf["num_classes"]):
    '''Returns mixup loss'''
    ys_onehot = [label_to_onehot(y, num_classes=num_class) for y in ys]
    mixy = vec_mul_ten(lam_batch[:, 0], ys_onehot[0])
    for i in range(1, klam):
        mixy += vec_mul_ten(lam_batch[:, i], ys_onehot[i])
    l = cross_entropy_for_onehot(pred, mixy)
    return l