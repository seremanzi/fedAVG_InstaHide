import json,os

import argparse
from conf import conf
import torch
import numpy as np
from fedavg.server import Server
from fedavg.client import Client

from fedavg.models import resnet20
from utils import get_data
import copy
import wandb
from fedavg.models import ReTrainModel

parser = argparse.ArgumentParser(
    description='PyTorch InstaHide Training, CIFAR-100')


# InstaHide configurations
parser.add_argument('--klam', default=4, type=int, help='number of lambdas')

args = parser.parse_args()

if __name__ == '__main__':

    wandb.init(project="", entity="samaml", group="fedavg_instahide", name="fedavg_ih")

    train_datasets, val_datasets, test_dataset = get_data()

    ### Initialize the aggregation weight of each node
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)

    print("Aggregation weight initialized")

    ## Save nodes
    clients = {}
    # Save node models
    clients_models = {}

    if conf['model_name'] == 'resnet20':
        model = resnet20(100)

    if torch.cuda.is_available():
        model.cuda()

    server = Server(conf, model, test_dataset)

    print("Server initialized!")

    for key in train_datasets.keys():
        clients[key] = Client(conf, server.global_model, train_datasets[key], val_datasets[key])

    print("Clients initialized!")

    # Save the model
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    max_acc = 0

    # Federated learning
    for e in range(conf["global_epochs"]):
        print("Epoch %d of %d" % (e,conf["global_epochs"]))

        for key in clients.keys():
            print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model, args.klam)
            clients_models[key] = copy.deepcopy(model_k)

        # Federated aggregation
        server.model_aggregate(clients_models, client_weight)
        # Test the global model
        acc, loss = server.model_eval()
        print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))

        # Save test accuracy to wandb
        wandb.log({"Test Accuracy": acc})

        # Save the best model
        if acc >= max_acc:
            torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"], "model-epoch{}.pth".format(e)))
            max_acc = acc

    torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"],conf["model_file"]))

    print("Federated training completed, the model is saved in the {0} directory!".format(conf["model_dir"]))