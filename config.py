""" Config class for search/augment """
import argparse
import os
from functools import partial
from configobj import ConfigObj
import torch
import logging


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--cfg', type=str)
        parser.add_argument('--name', type=str)
        parser.add_argument('--validation_top_k', type=int, default=5)                
        parser.add_argument('--controller_class', type = str, default = 'models.search_cnn.SearchCNNController' )
        parser.add_argument('--dataset',  type=str, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--validate_split', type=float, default=0.5, help='Split into train/test part (0 for run without validation part)')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--simple_alpha_update', type=int, default=0, help='update alpha without unrolling')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')

        return parser

    def __init__(self, section='train'):
        parser = self.build_parser()
        args = parser.parse_args()
        if args.cfg is not None:
            logging.debug('external config found')
            cfg = ConfigObj(args.cfg)
            for k in cfg[section]:
                if k in args.__dict__ and args.__dict__[k]:
                    t = type(args.__dict__[k]) 
                    if not t:
                        t = str
                else:
                    t = str    
                                                     
                args.__dict__[k] = t(cfg[section][k])
        if  not args.name or not args.dataset:
            raise ValueError()                            
        
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.fine_tune_path = os.path.join(self.path, 'fine_tune')
        self.gpus = parse_gpus(self.gpus)


