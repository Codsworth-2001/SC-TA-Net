import argparse
from torchvision import transforms
import logging, os, time
from net_model import NetModel
from dataset import MyDataset
import collections


def get_args_parser():
    parser = argparse.ArgumentParser('Image recognition training(and evaluation) script', add_help=False)
    parser.add_argument('--batch_size',  type=int, help='batch size when the model receive data')
    parser.add_argument('--epochs',  type=int)
    parser.add_argument('--update_freq',  type=int, help='gradient accumulation steps')

    parser.add_argument('--model',  type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--dataset',  type=str,  help='Name of dataset')
    parser.add_argument('--input_size',  type=int, help='input size or dim of the data feed into the model')
    parser.add_argument('--hidden_size',  type=int, help='hidden_size of the rnn based model')
    parser.add_argument('--channel_dim',  type=int, help='channel dim of the cnn based model')
    parser.add_argument('--channel_dim2',  type=int, help='channel dim of the cnn based model')
    parser.add_argument('--num_layer',  type=int, help='Number of the lstm cell layers in the model')
    parser.add_argument('--num_heads',  type=int, help='Number of the heads in multihead attention when using transformer series')
    parser.add_argument('--dropout',  type=float, help='Probabilities of the drop rate in the dropout layer')
    parser.add_argument('--num_classes',  type=int, help='Number of classes')
    parser.add_argument('--num_channels', type=int, help='Number of channels')
    parser.add_argument('--fs',  type=int, help='Number of classes')
    parser.add_argument('--subnum',  type=int, help='Number of sub')

    parser.add_argument('--opt', default='sgd', type=str, metavar='Optimizer', help='Optimizer (default: "sgd/adaw"')
    parser.add_argument('--criterion',  type=str, metavar='criterion',
                        help='criterion (default: "cross entropy"')
    parser.add_argument('--opt_eps',  type=float, metavar='Epsilon',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=list, nargs='+', metavar='Beta',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='Norm',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 0.05)')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--lr', type=float,  metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--min_lr', type=float, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--kfold', type=bool, default=False, help='whether to use kfold')
    parser.add_argument('--k', type=int, default=10, help='number of k in kfold,valid only when kfold=True')
    parser.add_argument('--num_patches', type=int, default=4)

    parser.add_argument('--data_path',  type=str, help='dataset path')

    # parser.add_argument('--subjects',
    #                     default=['1', '2'], type=list,
    #                     help='subjects used to build the dataset')
    parser.add_argument('--data_range', type=int,
                        help='range of the scalared data')
    parser.add_argument('--sample_size',  type=int,
                        help='sample size used to build the dataset')
    parser.add_argument('--checkpoint_path', default='./checkpoint_path', type=str,
                        help='checkpoint path')
    parser.add_argument('--log_dir', default='./Log', help='path where to save log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--k_length',  type=int)
    parser.add_argument('--k_length2', type=int)
    parser.add_argument('--subjects',default=[  '1', '2', '3',
                                                ], type=list,)
    return parser


def main(args):

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    date_and_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_filename = os.path.join(args.log_dir, str(args.model)+'.log')
    if os.path.exists(log_filename): os.remove(log_filename)
    logger = logging.getLogger()
    file_hanlder = logging.FileHandler(log_filename)
    console = logging.StreamHandler()

    file_hanlder.setLevel(logging.INFO)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    file_hanlder.setFormatter(formatter)
    console.setFormatter(formatter)
    logger.addHandler(file_hanlder)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

    logger.info(str(args))
    accuracy = []
    for sub in args.subjects:
        accc= []
        num = 0
        args.subnum = int(sub)-1

        alldataset = MyDataset(args.sample_size, args.dataset,0, sub=sub, sam=0,transform=None)
        args.seed = args.seed + 1
        print(collections.Counter(alldataset.label))
        model = NetModel(args)
        model.set_dataset(alldataset)
        acc_list, precision_list, recall_list, F1_list = model.train_with_Kfold(logger, alldataset, sub, args.k,True)

        with open('result_' + args.dataset + '/result' + args.model +'.txt', 'a') as file:
            acc = sum(acc_list)/len(acc_list)
            accc.append(acc)

            recall = sum(recall_list)/len(recall_list)
            precision = sum(precision_list) / len(precision_list)
            F1 = sum(F1_list) / len(F1_list)
            file.write('channel' + sub + ': acc: ')
            file.write(str(acc))
            file.write(' | ')
            for item in acc_list:
                file.write(str(item)+' ')
            file.write('\n')
            file.write('precision:')
            file.write(str(precision))
            file.write(' | ')
            for item in precision_list:
                file.write(str(item) + ' ')
            file.write('\n')
            file.write('recall:')
            file.write(str(recall))
            file.write(' | ')
            for item in recall_list:
                file.write(str(item) + ' ')
            file.write('\n')
            file.write('F1:')
            file.write(str(F1))
            file.write(' | ')
            for item in F1_list:
                file.write(str(item) + ' ')
            file.write('\n')
        for h in range(len(accc)):
            num = num+accc[h]
        accuracy.append((num/len(accc),sub))
    sorted_acc = sorted(accuracy, key= lambda x: x[0], reverse=True)
    with open('result_' + args.dataset + '/result' + args.model + '.txt', 'a') as file:
        file.write('Fre+Tem highest acc: ')
        for i in range(5):
            file.write(sorted_acc[i][1]+' ')
        file.write('\n')
        file.write(str(args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image recognition training(and evaluation) script', parents=[get_args_parser()])

    # parser = get_args_parser()
    arglist = ['--model', 'SCTANet',
               '--dataset', '',
               '--input_size', '200',
               '--num_patches','1',
               '--num_channels', '',
               '--dropout', '0.5',
               '--num_classes', '',
               '--fs', '200',
               '--hidden_size','',
               '--num_layer','',
               '--sample_size', '',
               '--batch_size', '',
               '--epochs', '50',
               '--seed', '',
               '--update_freq', '1',
               '--kfold', 'True', '--k', '5',
               '--opt', 'adamw', '--opt_eps', '1e-8',
               '--clip_grad', '1',
               '--momentum', '0.9',
               '--weight_decay', '0.005',
               '--lr', '5e-4', '--min_lr', '1e-6',
               '--criterion', 'cross entropy']

    args = parser.parse_args(args=arglist)
    main(args)

