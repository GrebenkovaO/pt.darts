""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_var_naive.search_cells import SearchCell
from models.cnn_var_naive.ops import NaiveVarConv2d, NaiveVarLinear
import genotypes as gt
from torch.nn.parallel._functions import Broadcast

from visualize import plot
import logging


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class VarSearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self,  C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            NaiveVarConv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur,
                              reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = NaiveVarLinear(C_p, n_classes)
        self.log_q_t = nn.Parameter(torch.zeros(1))
        self.q_gamma_normal = nn.ParameterList()
        self.q_gamma_reduce = nn.ParameterList()
        n_ops = len(gt.PRIMITIVES)

        for i in range(n_nodes):
            self.q_gamma_normal.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.q_gamma_reduce.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        self.stochastic_gamma = True
        self.stochastic_w = True

    def forward(self, x):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            gammas = self.q_gamma_reduce if cell.reduction else self.q_gamma_normal
            if self.stochastic_gamma:
                weights = [torch.distributions.RelaxedOneHotCategorical(
                    torch.exp(self.log_q_t), logits=gamma).rsample() for gamma in gammas]
            else:
                weights = gammas
                

            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def disable_stochastic_gamma(self):
        self.stochastic_gamma = False

    def disable_stochastic_w(self):
        self.stochastic_w = False
        all_ = [self]
            i = 0 
            while i<len(all_):
                current = all_[i]
                i+=1
                try:
                    for c in current.children():
                        all_+=[c]
                except:
                    pass
            for c in all_:
                if 'stochastic' in c.__dict__:
                    c.stochastic = False  


    def prune(self, k=2):        
        self.stochastic = False 
        for edges in self.q_gamma_normal:           
            edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
            edges.data*=0
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
            node_gene = []
            
            for edge_idx in topk_edge_indices:
                edges.data[edge_idx, primitive_indices[edge_idx]] += 1
        for edges in self.q_gamma_reduce:           
            edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
            edges.data*=0
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
            node_gene = []
            for edge_idx in topk_edge_indices:
                edges.data[edge_idx, primitive_indices[edge_idx]] += 1
        for c in self.cells:
            for subdag in c.dag:
                for mix in subdag:                
                    for op in mix._ops:
                        op.stochastic = False 
        self.linear.stochastic = False 


                

class VarSearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, device, **kwargs):
        super().__init__()
        C_in = int(kwargs['input_channels'])
        C = int(kwargs['init_channels'])
        n_classes = int(kwargs['n_classes'])
        n_layers = int(kwargs['layers'])
        n_nodes = int(kwargs['n_nodes'])
        stem_multiplier = int(kwargs['stem_multiplier'])
        device_ids = kwargs.get('device_ids', None)
        self.dataset_size = int(kwargs['dataset size'])
        self.n_nodes = n_nodes
        self.stochastic = True
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.criterion = nn.CrossEntropyLoss().to(device)

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)
        self.t_h = nn.Parameter(torch.ones(1) * float(kwargs['initial temp']))

        self.delta = float(kwargs['delta'])

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        self.alpha_w_h = {}
        self.alpha_h = nn.ParameterList()

        self.sample_num = int(kwargs['sample num'])

        self.net = VarSearchCNN(C_in, C,  n_classes,
                                n_layers, n_nodes, stem_multiplier)

        for w in self.net.parameters():
            if 'sigma' in w.__dict__:
                self.alpha_h.append(nn.Parameter(torch.zeros(w.shape)))
                self.alpha_w_h[w] = self.alpha_h[-1]

        for i in range(n_nodes):
            self.alpha_normal.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        
        self.stochastic_gamma =  int(kwargs['stochastic_gamma'])!=0
        self.stochastic_w =  int(kwargs['stochastic_w'])!=0
        if not self.stochastic_w:
            self.net.disable_stochastic_w()

    def new_epoch(self):
        self.t_h.data += self.delta

    def forward(self, x):
        return self.net(x)

    def loss(self, X, y):
        logits = self.forward(X)
        if self.stochastic:
            kld = self.kld()
            #self.t_h.data += self.delta
            return kld / self.dataset_size + self.criterion(logits, y)
        else:
            return self.criterion(logits, y)

    def kld(self):
        k = 0
        if self.stochastic_w:
            for w, h in self.alpha_w_h.items():
                eps_w = torch.distributions.Normal(w, torch.exp(w.sigma))
                eps_h = torch.distributions.Normal(w*0, torch.exp(h))
                k += torch.distributions.kl_divergence(eps_w, eps_h).sum()

        for a, ga in zip(self.alpha_normal, self.net.q_gamma_normal):
            g = torch.distributions.RelaxedOneHotCategorical(
                torch.exp(self.net.log_q_t), logits=ga)
            p_g = torch.distributions.RelaxedOneHotCategorical(
                self.t_h, logits=a)
            for _ in range(self.sample_num):
                sample = (g.rsample()+0.0001)
                k += (g.log_prob(sample) - p_g.log_prob(sample)).sum() / \
                    self.sample_num

        for a, ga in zip(self.alpha_reduce, self.net.q_gamma_reduce):
            g = torch.distributions.RelaxedOneHotCategorical(
                torch.exp(self.net.log_q_t), logits=ga)
            p_g = torch.distributions.RelaxedOneHotCategorical(
                self.t_h, logits=a)

            for _ in range(self.sample_num):
                sample = (g.rsample()+0.0001)
                k += (g.log_prob(sample) - p_g.log_prob(sample)).sum() / \
                    self.sample_num
        return k

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))
        if self.stochastic:
            logger.info("####### ALPHA #######")
            logger.info("# Alpha - normal")
            for alpha in self.alpha_normal:
                logger.info(F.softmax(alpha, dim=-1))

            logger.info("\n# Alpha - reduce")
            for alpha in self.alpha_reduce:
                logger.info(F.softmax(alpha, dim=-1))
            logger.info("#####################")

            logger.info("####### GAMMA #######")
            logger.info("# Gamma - normal")
            
            for alpha in self.net.q_gamma_normal:
                logger.info(F.softmax(alpha, dim=-1))

            logger.info("\n# Gamma - reduce")
            for alpha in self.net.q_gamma_reduce:
                logger.info(F.softmax(alpha, dim=-1))
            logger.info("#####################")

            logger.info('Temp:'+str(torch.exp(self.net.log_q_t)))
        else:
            logger.info("####### GAMMA #######")
            logger.info("# Gamma - normal")
            
            for alpha in self.net.q_gamma_normal:
                logger.info(alpha)

            logger.info("\n# Gamma - reduce")
            for alpha in self.net.q_gamma_reduce:
                logger.info(alpha)
            logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def prune(self):        
        self.stochastic = False
        self.net.stochastic = False 
        self.net.prune()
        
    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def plot_genotype(self, plot_path, caption):
        plot(self.genotype().normal, plot_path+'-normal', caption+'-normal')
        plot(self.genotype().reduce, plot_path+'-reduce', caption+'-reduce')
