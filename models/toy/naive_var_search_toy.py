""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn import Linear, Parameter
import logging
import torch as t

logger = logging.getLogger('darts')
sigma_init = -2.0
h_sigma_init = -0.0

class NVSearhToy(nn.Module):
    """ Search CNN model """
    def __init__(self):
        super().__init__()
        self.model1_m = nn.Parameter(t.ones(2))
        
        self.model2_m = nn.Parameter(t.ones(1,2))
        t.nn.init.xavier_uniform(self.model2_m)
        
        self.model3_m = nn.Parameter(t.ones(25,2))
        t.nn.init.xavier_uniform(self.model3_m)
        
        self.model_sm_m = nn.Parameter(t.ones(2,2))
        t.nn.init.xavier_uniform(self.model_sm_m)        

        self.model1_s = nn.Parameter(t.ones(2)*sigma_init)
        
        self.model2_s = nn.Parameter(t.ones(1,2)*sigma_init)
        
        self.model3_s = nn.Parameter(t.ones(25,2)*sigma_init)

        self.model_sm_s = nn.Parameter(t.ones(2,2) * sigma_init)

        
        self.stochastic = True


    def forward(self, x,  alpha):
        if self.stochastic:
            eps_1 = t.distributions.Normal(self.model1_m, t.exp(self.model1_s))
            eps_2 = t.distributions.Normal(self.model2_m, t.exp(self.model2_s))            
            eps_3 = t.distributions.Normal(self.model3_m, t.exp(self.model3_s))
            eps_sm = t.distributions.Normal(self.model_sm_m, t.exp(self.model_sm_s))            
            r1 = F.tanh(eps_1.rsample())
            r2 = F.tanh(t.matmul(x[:,:1], eps_2.rsample()))
            r3 = F.tanh(t.matmul(x, eps_3.rsample()))
            logits = t.matmul(r1*alpha[0]+r2*alpha[1]+r3*alpha[2], eps_sm.rsample())
        else:
            r1 = F.tanh(self.model1_m)
            r2 = F.tanh(t.matmul(x[:,:1], self.model2_m))
            r3 = F.tanh(t.matmul(x, self.model3_m))
            logits = t.matmul(r1*alpha[0]+r2*alpha[1]+r3*alpha[2], self.model_sm_s)

        return logits
        
    def kld(self, model1_h, model2_h, model3_h, model_sm_h):
        eps_1 = t.distributions.Normal(self.model1_m, t.exp(self.model1_s))
        eps_2 = t.distributions.Normal(self.model2_m, t.exp(self.model2_s))            
        eps_3 = t.distributions.Normal(self.model3_m, t.exp(self.model3_s))
        eps_sm = t.distributions.Normal(self.model_sm_m, t.exp(self.model_sm_s))                    
        p_1 = t.distributions.Normal(self.model1_m * 0 , t.exp(model1_h))
        p_2 = t.distributions.Normal(self.model2_m * 0 , t.exp(model2_h))
        p_3 = t.distributions.Normal(self.model3_m * 0 , t.exp(model3_h))        
        p_sm = t.distributions.Normal(self.model_sm_m * 0, t.exp(model_sm_h))                            
        return t.distributions.kl_divergence(eps_1, p_1).sum()+ t.distributions.kl_divergence(eps_2, p_2).sum() +t.distributions.kl_divergence(eps_sm, p_sm).sum()  
        

class NVSearchToyController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self,  device, **kwargs):
        super().__init__()
        self.net = NVSearhToy()
                
        self.model1_h = nn.Parameter(t.ones(2)*h_sigma_init)
        
        self.model2_h = nn.Parameter(t.ones(1,2)*h_sigma_init)
        
        self.model3_h = nn.Parameter(t.ones(25,2)*h_sigma_init)
        
        self.model_sm_h = nn.Parameter(t.ones(2,2)*h_sigma_init)

        self.alphas_ = [nn.Parameter(1e-3*torch.randn(3).to(device)), self.model1_h, self.model2_h, self.model3_h, self.model_sm_h]
        self.pruned = False
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.dataset_size = int(kwargs['dataset size'])


    def forward(self, x):
        if not self.pruned:
            weights_normal = F.softmax(self.alphas_[0], dim=-1)
        else:
            weights_normal = self.alphas_[0]
        return self.net(x, weights_normal)


    def loss(self, X, y):
        logits = self.forward(X)
    
        if self.net.stochastic:
            kld = self.net.kld(self.alphas_[1], self.alphas_[2], self.alphas_[3], self.alphas_[4])

            return kld / self.dataset_size  + self.criterion(logits, y) 
        else:
            return self.criterion(logits, y) 
            

    def alphas(self):
        return self.alphas_

    def print_alphas(self, logger):
        
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        if not self.pruned:
            logger.info(F.softmax(self.alphas_[0], dim=-1))
        else:
            logger.info(self.alphas_[0])


        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    
    def weights(self):
        return self.net.parameters()

    def genotype(self):
        names = ['const', 'simple' ,'complex']
        return names[self.alphas()[0].argmax()]
    
    def prune(self):
        argmax = self.alphas()[0].argmax()
        self.alphas_[0].data *= 0 
        self.alphas_[0].data[argmax] += 1
        self.pruned = True
        
    def plot_genotype(self,path,caption):
        pass 
            


class NVGSSearchToy(nn.Module):
    """ Search CNN model """
    def __init__(self, sample_num):

        super().__init__()
        self.model1_m = nn.Parameter(t.ones(2))
        
        self.model2_m = nn.Parameter(t.ones(1,2))
        t.nn.init.xavier_uniform(self.model2_m)
        
        self.model3_m = nn.Parameter(t.ones(25,2))
        t.nn.init.xavier_uniform(self.model3_m)
        
        self.model_sm_m = nn.Parameter(t.ones(2,2))
        t.nn.init.xavier_uniform(self.model_sm_m)        

        self.model1_s = nn.Parameter(t.ones(2)*sigma_init)
        
        self.model2_s = nn.Parameter(t.ones(1,2)*sigma_init)
        
        self.model3_s = nn.Parameter(t.ones(25,2)*sigma_init)

        self.model_sm_s = nn.Parameter(t.ones(2,2) * sigma_init)
        
        

        self.stochastic = True
        
        self.gamma = nn.Parameter(t.ones(3))
        
        self.log_temp = nn.Parameter(t.zeros(1))

        self.sample_num = sample_num
        


    def forward(self, x):
        if self.stochastic:
            g = t.distributions.RelaxedOneHotCategorical(t.exp(self.log_temp), logits=self.gamma)        
            alpha = g.rsample()+0.05
            
            eps_1 = t.distributions.Normal(self.model1_m, t.exp(self.model1_s))
            eps_2 = t.distributions.Normal(self.model2_m, t.exp(self.model2_s))            
            eps_3 = t.distributions.Normal(self.model3_m, t.exp(self.model3_s))
            eps_sm = t.distributions.Normal(self.model_sm_m, t.exp(self.model_sm_s))            
            r1 = F.tanh(eps_1.rsample())
            r2 = F.tanh(t.matmul(x[:,:1], eps_2.rsample()))
            r3 = F.tanh(t.matmul(x, eps_3.rsample()))
            logits = t.matmul(r1*alpha[0]+r2*alpha[1]+r3*alpha[2], eps_sm.rsample())
            
        else:
            alpha = self.determined_alpha
            r1 = F.tanh(self.model1_m)
            r2 = F.tanh(t.matmul(x[:,:1], self.model2_m))
            r3 = F.tanh(t.matmul(x, self.model3_m))
            logits = t.matmul(r1*alpha[0]+r2*alpha[1]+r3*alpha[2], self.model_sm_s)    

        return logits
        
    def kld(self, model1_h, model2_h, model3_h, model_sm_h, gamma_h, gamma_t):
        eps_1 = t.distributions.Normal(self.model1_m, t.exp(self.model1_s))
        eps_2 = t.distributions.Normal(self.model2_m, t.exp(self.model2_s))            
        eps_3 = t.distributions.Normal(self.model3_m, t.exp(self.model3_s))
        eps_sm = t.distributions.Normal(self.model_sm_m, t.exp(self.model_sm_s))                    
        p_1 = t.distributions.Normal(self.model1_m * 0 , t.exp(model1_h))
        p_2 = t.distributions.Normal(self.model2_m * 0 , t.exp(model2_h))
        p_3 = t.distributions.Normal(self.model3_m * 0 , t.exp(model3_h))        
        p_sm = t.distributions.Normal(self.model_sm_m * 0, t.exp(model_sm_h))        
        kld_w =  t.distributions.kl_divergence(eps_1, p_1).sum()+ t.distributions.kl_divergence(eps_2, p_2).sum() +t.distributions.kl_divergence(eps_3, p_3).sum() +t.distributions.kl_divergence(eps_sm, p_sm).sum()        
        
        g = t.distributions.RelaxedOneHotCategorical(t.exp(self.log_temp), logits=self.gamma)        
        p_g = t.distributions.RelaxedOneHotCategorical(gamma_t, logits=gamma_h)
        
        kld_h = 0
        for _ in range(self.sample_num):
            sample = (g.rsample()+0.0001)
            kld_h += (g.log_prob(sample) - p_g.log_prob(sample))/self.sample_num

        return kld_w + kld_h
                        

    def prune(self):
        """
        eps_1 = t.distributions.Normal(self.model1_m, t.exp(self.model1_s))
        eps_2 = t.distributions.Normal(self.model2_m, t.exp(self.model2_s))            
        eps_3 = t.distributions.Normal(self.model3_m, t.exp(self.model3_s))                
        g = t.distributions.RelaxedOneHotCategorical(t.exp(self.log_temp), logits=self.gamma)        
        
        simplex = t.zeros(3).to(self.model1_m.device)
        simplex.data[0]+=0.99        
        simplex.data[1]+=0.005
        simplex.data[2]+=0.005
                
        proba_1 = eps_1.log_prob(self.model1_m).sum() + eps_2.log_prob(self.model2_m*0).sum() + eps_3.log_prob(self.model3_m*0).sum() + g.log_prob(simplex).sum()
        logger.info('proba for const: {}'.format(proba_1)) 

        simplex.data*=0           
        simplex.data[0]+=0.005
        simplex.data[1]+=0.99             
        simplex.data[2]+=0.005
        
        proba_2 = eps_1.log_prob(self.model1_m*0).sum() + eps_2.log_prob(self.model2_m).sum() + eps_3.log_prob(self.model3_m*0).sum() + g.log_prob(simplex).sum()
        logger.info('proba for simple: {}'.format(proba_2))
        
        
        simplex.data*=0            
        simplex.data[0]+=0.005
        simplex.data[1]+=0.005
        simplex.data[2]+=0.99            
        proba_3 = eps_1.log_prob(self.model1_m*0).sum() + eps_2.log_prob(self.model2_m * 0).sum() + eps_3.log_prob(self.model3_m).sum() + g.log_prob(simplex).sum()
        logger.info('proba for complex: {}'.format(proba_3))

        self.determined_alpha = t.zeros(3).to(self.model1_m.device)       
        if proba_1 >= proba_2 and proba_1>= proba_3:
            self.determined_alpha.data[0]+=1
            logger.info('selecting const model')
        elif proba_2 >= proba_3 and proba_2 >= proba_1:
             self.determined_alpha.data[1]+=1
             logger.info('selecting simple model')
        else:
            self.determined_alpha.data[2]+=1
            logger.info('selecting complex model')            
        self.stochastic = True
        """
        self.determined_alpha = t.zeros(3).to(self.model1_m.device)       
        self.determined_alpha.data[self.gamma.argmax()]+=1
        self.stochastic = True
        


class NVGSSearchToyController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self,  device, **kwargs):
        super().__init__()
        self.net = NVGSSearchToy(int(kwargs['sample num']))

        
                
        self.model1_h = nn.Parameter(t.ones(2)*h_sigma_init)
        
        self.model2_h = nn.Parameter(t.ones(1,2)*h_sigma_init)
        
        self.model3_h = nn.Parameter(t.ones(25,2)*h_sigma_init)
        
        self.model_sm_h = nn.Parameter(t.ones(2,2)*h_sigma_init)
        
        self.gamma_h = nn.Parameter(t.ones(3))
        
        self.t_h = nn.Parameter(t.ones(1) * float(kwargs['initial temp']))
        
        self.delta = float(kwargs['delta'])
        
        self.dataset_size = int(kwargs['dataset size'])


        self.alphas_ = [nn.Parameter(1e-3*torch.randn(3).to(device)), self.model1_h, self.model2_h, self.model3_h, self.model_sm_h, self.gamma_h]
        self.pruned = False
        self.criterion = nn.CrossEntropyLoss().to(device)
        



    def forward(self, x):
        
        return self.net(x)


    def loss(self, X, y):
        logits = self.forward(X)
        if self.net.stochastic:
            kld = self.net.kld(self.alphas_[1], self.alphas_[2], self.alphas_[3], self.alphas_[4],  self.alphas_[5], self.t_h)
            self.t_h.data += self.delta

            return kld/ self.dataset_size  + self.criterion(logits, y)
        else:
            return self.criterion(logits, y)
            

    def alphas(self):
        return self.alphas_

    def print_alphas(self, logger):
        
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        if not self.pruned:
            logger.info("####### ALPHA #######")
            logger.info(F.softmax(self.alphas_[5], dim=-1))
            logger.info(self.t_h)            
            logger.info("####### GAMMA #######")
            logger.info(F.softmax(self.net.gamma, dim=-1))
            logger.info(t.exp(self.net.log_temp))
        else:            
            logger.info("####### GAMMA #######")
            logger.info(self.net.gamma)
                                                        

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    
    def weights(self):
        return self.net.parameters()

    def genotype(self):
        names = ['const', 'simple' ,'complex']
        print ('ARGMAX', self.net.gamma.argmax())
        return names[self.net.gamma.argmax()]
    
    def prune(self):
        self.pruned = True
        self.net.prune()
        
    def plot_genotype(self,path,caption):
        pass 
            




