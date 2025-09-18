import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from engine import train_one_epoch, evaluate
import time, datetime
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,Normalizer
from copy import deepcopy
import collections

from SCTANet import SCTANet

class NetModel(object):
    def __init__(self, args):
        self.args = args
        self.seed_all()
        self.set_model()

    def seed_all(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)

    def set_dataset(self, train_set, test_set=None):
        self.normalizer = MinMaxScaler(feature_range=(-1*self.args.sample_size,self.args.sample_size))
        #self.normalizer = Normalizer()
        self.train_set = train_set
        print(collections.Counter(self.train_set.label))
        if test_set is not None:
            self.test_set = test_set
            print(collections.Counter(self.test_set.label))
            self.train_set.dataset = self.normalizer.fit_transform(self.train_set.dataset)
            self.test_set.dataset = self.normalizer.transform(self.test_set.dataset)
            self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.args.num_workers, pin_memory=self.args.pin_mem)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers, pin_memory=self.args.pin_mem)
        self.total_batch_size = self.batch_size * self.args.update_freq
        self.num_training_steps_per_epoch = len(train_set) // self.total_batch_size
        self.scheduler = self.get_scheduler()
    def restore_data_range(self):
        self.train_set.dataset.dataset = self.normalizer.inverse_transform(self.train_set.dataset.dataset)
        if self.test_set is not None:
            self.test_set.dataset.dataset = self.normalizer.inverse_transform(self.test_set.dataset.dataset)
    def set_model(self):
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.epochs = self.args.epochs
        self.num_classes = self.args.num_classes
        self.clip_grad = self.args.clip_grad
        self.checkpoint_path = self.args.checkpoint_path
        self.update_freq = self.args.update_freq
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        self.network = self.get_network().to(self.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def train_with_Kfold(self,logger,alldataset, Sub , k = 10,test_after_one_epoch=False,):
        logger.info('number of params:%d' % self.n_parameters)
        logger.info("LR = %.8f" % self.lr)
        logger.info("Batch size = %d" % self.total_batch_size)
        logger.info("Update frequent = %d" % self.args.update_freq)
        logger.info("Number of training examples = %d" % len(self.train_set))
        logger.info("Number of training training per epoch = %d" % self.num_training_steps_per_epoch)
        logger.info("criterion = %s" % str(self.criterion))
        logger.info("scheduler = %s" % str(self.scheduler))
        logger.info("Start training for %d Folds" % self.args.k)
        loss_list = []
        acc_list = []
        precision_list = []
        recall_list = []
        F1_list = []
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.args.seed)
        for i,(train_index, test_index) in enumerate(kfold.split(alldataset.dataset,alldataset.label)):
            # train_fold = torch.utils.data.dataset.Subset(alldataset,train_index)
            # test_fold = torch.utils.data.dataset.Subset(alldataset,test_index)
            train_fold = deepcopy(alldataset)
            train_fold.dataset = train_fold.dataset[train_index]
            train_fold.label = train_fold.label[train_index]
            test_fold = deepcopy(alldataset)
            test_fold.dataset = test_fold.dataset[test_index]
            test_fold.label = test_fold.label[test_index]
            self.set_dataset(train_fold,test_fold)#reset dataset when using kfold
            logger.info("Start training for %d epochs with Fold %d" % (self.epochs,i))
            '''
            self.network.reset_parameters()
            self.scheduler._initial_step()
            '''
            self.network = self.get_network().to(self.device)
            self.optimizer = self.get_optimizer()
            self.scheduler = self.get_scheduler()

            start_time = time.time()
            for epoch in range(0, self.epochs):
                train_stats = train_one_epoch(model=self.network, criterion=self.criterion,
                                              data_loader=self.train_loader,
                                              optimizer=self.optimizer, device=self.device, epoch=epoch,
                                              lr_schedule=self.scheduler, logger=logger,
                                              clip_grad=self.clip_grad,
                                              num_training_steps_per_epoch=self.num_training_steps_per_epoch,
                                              update_freq=self.update_freq)
                if test_after_one_epoch:
                    test_stats = self.test(logger)
                if epoch == self.epochs - 1:
                    self.save_params(checkpoint_path='./checkpoint_path', epoch=epoch + 1, sub=Sub)
            #self.restore_data_range()
            test_stats = self.test(logger)
            loss_list.append(test_stats['avg loss'])
            acc_list.append(test_stats['acc'])
            precision_list.append(test_stats['precision'])
            recall_list.append(test_stats['recall'])
            F1_list.append(test_stats['F1'])

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.info('Training time {}'.format(total_time_str))
        return acc_list, precision_list, recall_list, F1_list


    def train(self, logger, test_after_one_epoch=False):
        logger.info('number of params:%d' % self.n_parameters)
        logger.info("LR = %.8f" % self.lr)
        logger.info("Batch size = %d" % self.total_batch_size)
        logger.info("Update frequent = %d" % self.args.update_freq)
        logger.info("Number of training examples = %d" % len(self.train_set))
        logger.info("Number of training training per epoch = %d" % self.num_training_steps_per_epoch)
        logger.info("criterion = %s" % str(self.criterion))
        logger.info("scheduler = %s" % str(self.scheduler))
        logger.info("Start training for %d epochs" % self.epochs)
        start_time = time.time()
        for epoch in range(0, self.epochs):
            train_stats = train_one_epoch(model=self.network, criterion=self.criterion, data_loader=self.train_loader,
                                          optimizer=self.optimizer, device=self.device, epoch=epoch,
                                          lr_schedule=self.scheduler, logger=logger,
                                          clip_grad=self.clip_grad,
                                          num_training_steps_per_epoch=self.num_training_steps_per_epoch,
                                          update_freq=self.update_freq)

            if test_after_one_epoch:
                test_stats = self.test(logger)

            # self.save_params(epoch=0,checkpoint_path='./checkpoint_path')
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))



    def test(self, logger):
        start_time = time.time()
        test_stats = evaluate(model=self.network, data_loader=self.test_loader, device=self.device, logger=logger, num_class=self.args.num_classes)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Testing time {}'.format(total_time_str))
        return test_stats

    def get_network(self):
        self.network_name = self.args.model

        if self.args.model == 'SCTANet':
            return SCTANet(num_class=self.num_classes, k=self.args.num_patches, query_channel_index=self.args.subnum, num_bands=5, num_channels=self.args.num_channels)

        else:
            raise Exception('Model Not Implementation!')

    def get_optimizer(self):
        opt_args = dict(lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.opt == 'sgd':
            return torch.optim.SGD(self.network.parameters(), momentum=self.args.momentum, **opt_args)
        elif self.args.opt == 'adam':
            return torch.optim.Adam(self.network.parameters(), betas=self.args.opt_betas, **opt_args)
        elif self.args.opt == 'adamw':
            return torch.optim.AdamW(self.network.parameters(), betas=self.args.opt_betas, **opt_args)
        else:
            raise Exception('Optimizer Not Implementation!')

    def get_criterion(self):
        if self.args.criterion == 'l1':
            return torch.nn.L1Loss()
        elif self.args.criterion == 'smooth l1':
            return torch.nn.SmoothL1Loss()
        elif self.args.criterion == 'mse':
            return torch.nn.MSELoss()
        elif self.args.criterion == 'cross entropy':
            return torch.nn.CrossEntropyLoss()
        elif self.args.criterion == 'nll':
            return torch.nn.NLLLoss()
        else:
            raise Exception('Criterion Not Implementation!')

    def get_scheduler(self):
        if self.args.scheduler == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= \
                self.epochs, eta_min=self.args.min_lr)
        elif self.args.schduler == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.args.scheduler == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        else:
            return None

    def save_params(self, epoch=0, checkpoint_path=None, sub=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_path = "{}/{}sub_{}_epoch_{}.pth".format(checkpoint_path, self.network_name, sub, epoch)
        torch.save(self.network.state_dict(), save_path)
        print("Save network parameters to {}".format(save_path))

    def load_params(self, epoch=0, checkpoint_path=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
            load_path = "{}/{}_epoch_{}.pth".format(checkpoint_path, self.network_name, epoch)
        else:
            load_path = checkpoint_path
        self.network.load_state_dict(torch.load(load_path, map_location=self.device))
        print("Load network parameters from {}".format(load_path))

    def load_feature_params_only(self, checkpoint_path):
        params = torch.load(checkpoint_path, map_location=self.device)
        remove_keys = []
        for key in params.keys():
            if "head" in key:
                print('removed head!')
                remove_keys.append(key)
        for key in remove_keys:
            params.__delitem__(key)
        self.network.load_state_dict(params, strict=False)
        print("Load network features parameters only from {}".format(checkpoint_path))
