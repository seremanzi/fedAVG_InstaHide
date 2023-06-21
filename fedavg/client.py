import torch
import numpy as np
from fedavg.datasets import get_dataset
from conf import conf
from torch.autograd import Variable
from utils import chunks, vec_mul_ten, mixup_criterion
import random


   
klam = 3


class Client(object):

    def __init__(self, conf, model, train_df, val_df):
        """
        :param conf: Configuration file
        :param model: Global model
        :param train_df: Training data DataFrame
        :param val_df: Validation data DataFrame
        """
        self.conf = conf

        self.local_model = model
        self.train_df = train_df
        self.train_dataset = get_dataset(conf, self.train_df)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],shuffle=True)

        self.val_df = val_df
        self.val_dataset = get_dataset(conf, self.val_df)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf["batch_size"],shuffle=True)
     
    
 
    
    def mixup_data(x, y, klam, use_cuda=True):
        device = torch.device("cuda" if use_cuda else "cpu")
        '''Returns mixed inputs, lists of targets, and lambdas'''
        lams = np.random.normal(0, 1, size=(x.size()[0], klam))
        for i in range(x.size()[0]):
            lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
            if klam > 1:
                while lams[i].max() > conf["upper"]:     # upper bounds a single lambda
                    lams[i] = np.random.normal(0, 1, size=(1, klam))
                    lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))

        lams = torch.from_numpy(lams).float().to(device)

        mixed_x = vec_mul_ten(lams[:, 0], x)
        ys = [y]

        for i in range(1, klam):
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(device)
            mixed_x += vec_mul_ten(lams[:, i], x[index, :])
            ys.append(y[index])

        
            sign = torch.randint(2, size=list(x.shape), device=device) * 2.0 - 1
            mixed_x *= sign.float().to(device)

        
        return mixed_x, ys, lams

    
    def generate_sample(self, trainloader, klam):
        use_cuda = torch.cuda.is_available()
        
        assert len(trainloader) == 1        # Load all training data once
        for _, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            mix_inputs, mix_targets, lams = self.mixup_data(
                inputs, targets.float(), klam, use_cuda)

        return (mix_inputs, mix_targets, lams)

        
    def train(net, optimizer, inputs_all, mix_targets_all, lams, klam):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
        #print('\nEpoch: %d' % epoch)
        #net.train()
        train_loss, correct, total = 0, 0, 0


        seq = random.sample(range(len(inputs_all)), len(inputs_all))
        bl = list(chunks(seq, ))

        for batch_idx in range(len(bl)):
            b = bl[batch_idx]
            inputs = torch.stack([inputs_all[i] for i in b])
            lam_batch = torch.stack([lams[i] for i in b])

            mix_targets = []
            for ik in range(klam):
                mix_targets.append(
                    torch.stack(
                        [mix_targets_all[ik][ib].long().to(device) for ib in b]))
            targets_var = [Variable(mix_targets[ik]) for ik in range(klam)]

            inputs = Variable(inputs)
            outputs = net(inputs)
            loss = mixup_criterion(outputs, targets_var, lam_batch, klam)
            train_loss += loss.data.item()
            total += conf["batch_size"]   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """
            progress_bar(batch_idx, len(inputs_all)/conf["batch_size"]+1,
                        'Loss: %.3f' % (train_loss / (batch_idx + 1)))
            """
        return (train_loss / batch_idx, 100. * correct / total)


    def test(net, optimizer, testloader, criterion):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
       
        global best_acc
        net.eval()
        test_loss, correct_1, correct_5, total = 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.data.item()
                _, pred = outputs.topk(5, 1, largest=True, sorted=True)
                total += targets.size(0)
                correct = pred.eq(targets.view(targets.size(0), -
                                            1).expand_as(pred)).float().cpu()
                correct_1 += correct[:, :1].sum()
                correct_5 += correct[:, :5].sum()
                """
                progress_bar(
                    batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (test_loss /
                        (batch_idx + 1), 100. * correct_1 / total, correct_1, total))
                """
        acc = 100. * correct_1 / total
        if acc > best_acc:
            best_acc = acc
        return (test_loss / batch_idx, 100. * correct_1 / total)





    def local_train(self, model, klam):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'],weight_decay=self.conf["weight_decay"])
        # optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.BCEWithLogitsLoss()
        
        
        for e in range(self.conf["local_epochs"]):
            self.local_model.train()
        
            mix_inputs_all, mix_targets_all, lams = self.generate_sample(self.train_loader, klam)
            
            train_loss, _ = self.train(self.local_model, optimizer, mix_inputs_all, mix_targets_all, lams, klam)
            test_loss, test_acc1, = self.test(self.local_model, optimizer, self.val_loader, criterion)
            

            """            
            for _, batch in enumerate(self.train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output, feature = self.local_model(data)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()
            """
        #acc, eval_loss = self.model_eval()
        #print("Epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(e, loss, eval_loss, acc))

        return self.local_model.state_dict()


    """
    def mixup_criterion(pred, ys, lam_batch, num_class=args.nclass):
        '''Returns mixup loss'''
        ys_onehot = [label_to_onehot(y, num_classes=num_class) for y in ys]
        mixy = vec_mul_ten(lam_batch[:, 0], ys_onehot[0])
        for i in range(1, args.klam):
            mixy += vec_mul_ten(lam_batch[:, i], ys_onehot[i])
        l = cross_entropy_for_onehot(pred, mixy)
        return l
    """
    @torch.no_grad()
    def model_eval(self):
        self.local_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.BCEWithLogitsLoss()
        for _, batch in enumerate(self.val_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output,feature = self.local_model(data)

            total_loss += criterion(output, target)    # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size

        return acc, total_l

    def _cal_mean_cov(self,features):

        features = np.array(features)
        mean = np.mean(features, axis=0)

        # Note that bias=1 is set here, which is equivalent to dividing by N in formula (2) instead of N-1,
        # because when N=1, dividing by 0 will result in NaN.
        cov = np.cov(features.T, bias=1)
        return mean,cov

    def cal_distributions(self, model):
        """
        :param feature:
        :return: Mean, covariance, and length
    """
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.eval()

        features = []
        mean = []
        cov = []
        length = []

        for i in range(self.conf["num_classes"]):
            train_i = self.train_df[self.train_df[self.conf['label_column']] == i]
            train_i_dataset = get_dataset(self.conf, train_i)

            if len(train_i_dataset) > 0:
                train_i_loader = torch.utils.data.DataLoader(train_i_dataset, batch_size=self.conf["batch_size"],
                                                             shuffle=True)
                for batch_id, batch in enumerate(train_i_loader):
                    data, target = batch

                    if torch.cuda.is_available():
                        data = data.cuda()

                    output, feature = self.local_model(data)
                    features.extend(feature.tolist())

                f_mean, f_cov = self._cal_mean_cov(features)

            else:
                # TODO: Determine the dimensions of mean and covariance based on the output of the last hidden layer
                f_mean = np.zeros((64,))
                f_cov = np.zeros((64,64))

            mean.append(f_mean)
            cov.append(f_cov)
            length.append(len(train_i))

        return mean, cov, length







