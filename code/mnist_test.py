"""
This code is for Experiment 1: MNIST
It trains and tests the best hyperparameter combination, found in grid_search.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import namedtuple
from namedlist import namedlist

def get_train_valid_loader(train_batch_size,
                           valid_batch_size,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Splits up the MNIST training set into a training set and a validation set.
    More precisely, loads and returns train and valid
    multi-process iterators over the MNIST dataset.
    Code adapted from:
    https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    
    normalize = transforms.Normalize((0,), (255./126.,)) #((0.1307,), (0.3081,)) # note 1

    # load the dataset
    train_dataset = datasets.MNIST(
        root='../data', train=True,
        download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalize]))

    valid_dataset = datasets.MNIST(
        root='../data', train=True,
        download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalize]))
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory)
    
    return (train_loader, valid_loader)
  
def get_test_loader(batch_size,
                   shuffle=True,
                   num_workers=4,
                   pin_memory=False):
    '''Loads MNST training data'''
    normalize = transforms.Normalize((0,), (255./126.,))
    return torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=False, transform=transforms.Compose([
                         transforms.ToTensor(),
                         normalize])),
            batch_size=batch_size, shuffle=True, # note 8
            num_workers=num_workers, pin_memory=pin_memory)

class MLP(nn.Module):
    def __init__(self, units_per_layer):
        super().__init__()
        self.fc1 = nn.Linear(28*28, units_per_layer)
        self.fc2 = nn.Linear(units_per_layer, units_per_layer)
        self.fc3 = nn.Linear(units_per_layer, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # dimensions should be batch size x 784
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        y_hat = self.fc3(h2)
        return F.log_softmax(y_hat, dim=1) # note 2

class Dropout_MLP(nn.Module):
    '''MLP with dropout on both hidden layers
       p=0.5, as per the original paper
    '''
    def __init__(self, units_per_layer):
        super().__init__()
        self.fc1 = nn.Linear(28*28, units_per_layer)
        self.fc2 = nn.Linear(units_per_layer, units_per_layer)
        self.fc3 = nn.Linear(units_per_layer, 10)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 28*28) # dimensions should be batch size x 784
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        y_hat = self.fc3(x)
        return F.log_softmax(y_hat, dim=1) # note 2

class GaussianReparam:
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        
    def sample(self):
        epsilon = torch.randn_like(self.rho) # "randn" samples from standard normal distr
        return self.mu + torch.log(1 + torch.exp(self.rho)) * epsilon
    
    def log_prob(self, x):
        mu = self.mu
        sigma = torch.log(1 + torch.exp(self.rho))
        return torch.distributions.normal.Normal(mu, sigma).log_prob(x).sum()
    
class ScaleMixtureGaussian:
    def __init__(self, pi, sigma_1, sigma_2):
        self.pi = pi
        self.gaussian_1 = torch.distributions.normal.Normal(0, sigma_1)
        self.gaussian_2 = torch.distributions.normal.Normal(0, sigma_2)
        
    def log_prob(self, x):
        prob1 = torch.exp(self.gaussian_1.log_prob(x)) # for some reason PyTorch doesn't have a pdf function so take exp(log_prob(.))
        prob2 = torch.exp(self.gaussian_2.log_prob(x))
        return torch.log(self.pi * prob1 + (1-self.pi) * prob2) # this will get summed across both dimensions

class BayesianLayer(nn.Module):
    """Creates a single layer i.e. weight matrix of a BNN"""
    def __init__(self, in_units, out_units, prior_form):
        super().__init__()
        self.in_units = in_units
        self.out_units = out_units
        # Weights (distribution to sample from)
        self.weight_mu = nn.Parameter(torch.Tensor(out_units, in_units).uniform_(-0.2, 0.2)) # borrowed initalisation from nitarshan
        self.weight_rho = nn.Parameter(torch.Tensor(out_units, in_units).uniform_(-5, -4))
        self.weight = GaussianReparam(self.weight_mu, self.weight_rho) # type: out_units x in_units matrix of GaussianReparam() objects (which we can sample from or find log_probs)
        # Biases (distribution to sample from)
        self.bias_mu = nn.Parameter(torch.Tensor(out_units).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_units).uniform_(-5, -4))
        self.bias = GaussianReparam(self.bias_mu, self.bias_rho)
        # Priors (_shared_ distribution to sample from)
        if len(prior_form) == 3:
            nl_sig1, nl_sig2, pi = prior_form
            self.weight_prior = ScaleMixtureGaussian(pi, math.exp(-nl_sig1), math.exp(-nl_sig2))
            self.bias_prior = ScaleMixtureGaussian(pi, math.exp(-nl_sig1), math.exp(-nl_sig2))
        else: # use standard guassian priors
            nl_sigma, = prior_form
            self.weight_prior = torch.distributions.normal.Normal(0, math.exp(-nl_sigma))
            self.bias_prior = torch.distributions.normal.Normal(0,  math.exp(-nl_sigma))
        # Initialise log probs of prior and variational posterior for this layer
        self.log_variational_posterior = 0
        self.log_prior = 0
        
    def forward(self, x, take_sample=True):
        """Do a forward pass by sampling from variational posterior
           Also update log probabilities for varational posterior and prior
           given the current sampled weights and biases
           (we will need theseSubsetRandomSampler to compute the loss function)
        """
        if take_sample or self.training: # maybe get rid of the self.training since if I call it when it's not training, I set take_sample to False => it's redundant and confusing?
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training: # (*)
            self.log_variational_posterior = self.weight.log_prob(weight).sum() + self.bias.log_prob(bias).sum()
            self.log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
        else:
            self.log_prior, self.log_variational_posterior = 0, 0 # not sure what's going on here. At test time do we not want log probs for var posterior and prior??
        return F.linear(x, weight, bias)

class BayesianNet(nn.Module):
    def __init__(self, units_per_layer, args, bayes_params):
        super().__init__()
        n_train_samples, KL_reweight, prior_form = bayes_params
        self.v_samples = n_train_samples
        self.batch_size = args.batch_size
        self.num_batches = args.num_batches_train
        self.do_KL_reweighting = KL_reweight
        self.num_categories = 10
        # input: 28x28 pixel images = 784 units
        # hidden: units_per_layer
        # output: 10 categories
        # prior_form gives params for Gaussian or Scale Mixture or both
        self.layer1 = BayesianLayer(28*28, units_per_layer, prior_form)
        self.layer2 = BayesianLayer(units_per_layer, units_per_layer, prior_form)
        self.layer3 = BayesianLayer(units_per_layer, 10, prior_form)
        
    def forward(self, x, take_sample=True): # note 5
        x = x.view(-1, 28*28) # dim: batch size x 784
        x = F.relu(self.layer1(x, take_sample))
        x = F.relu(self.layer2(x, take_sample))
        logits = self.layer3(x, take_sample)
        y_hat = F.log_softmax(logits, dim=1)
        return y_hat
    
    def log_prior(self):
        '''log probability of the current prior parameters is the sum
           of those parameters for each layer.
           These get updated here (*) each time we do a forward pass
           This implies forward() must be called before finding the log lik
           of the posterior and prior parameters (in the loss func)!
        '''
        return self.layer1.log_prior \
            + self.layer2.log_prior \
            + self.layer3.log_prior
    
    def log_variational_posterior(self):
        '''log probability of the current posterior parameters is the sum
            of those parameters for each layer
        '''
        return self.layer1.log_variational_posterior + \
               self.layer2.log_variational_posterior + \
               self.layer3.log_variational_posterior
      
    def loss(self, input, target, batch_idx, device):
        """Variational free energy/negative ELBO loss function, called
           f(w, theta) in the paper
           NB calling model.loss() does a forward pass, so in train() function
           we don't need to call model(input)
        """
        outputs = torch.zeros(self.v_samples, self.batch_size, self.num_categories).to(device) # create tensors on the GPU
        log_priors = torch.zeros(self.v_samples).to(device)
        log_variational_posteriors = torch.zeros(self.v_samples).to(device)
        for i in range(self.v_samples):
            outputs[i] = self(input) # note 4
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_priors[i] = self.log_prior()
        log_variational_posterior = log_variational_posteriors.mean()
        log_prior = log_priors.mean()
        # the following line might be wrong -- change it if something's goin wrong
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction='sum') # we want to sum over (y_i.\hat{y_i}); y_i = 0 for all units except true output. Pretty sure this is identical to size_average=False
        if self.do_KL_reweighting:
            minibatch_weight = 1. / (2**self.num_batches - 1) * (2**(self.num_batches - batch_idx))
            loss = minibatch_weight * (log_variational_posterior - log_prior) + negative_log_likelihood
        else:
            loss = 1/self.num_batches * (log_variational_posterior - log_prior) + negative_log_likelihood
        return loss # they also return log_prior, log_variational_posterior, negative_log_likelihood

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train() # note 3
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # note 6
        if isinstance(model, (MLP, Dropout_MLP)):
            output = model(data) # note 4
            loss = F.nll_loss(output, target)
        elif isinstance(model, BayesianNet):
            loss = model.loss(data, target, batch_idx+1, device) # +1 for indices {1,..,M}
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(data) * len(train_loader)), # len(data) is size of train batch
                100. * batch_idx / len(train_loader), loss.item())) # len(train_loader) is number of train batches

def test(model, device, test_loader, epoch, test_batch_size, *args):
    '''*args captures the extra arguments given to test_ensemble that we don't
       need here
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # put tensors on GPU
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= (test_batch_size * len(test_loader)) # batch_size * num_test_batches
    test_accuracy = 100. * correct / (test_batch_size * len(test_loader))
    print('Validation set\nEpoch: {}\tAverage loss: {:.4f}\t Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, test_batch_size * len(test_loader), test_accuracy))
    error = 100 - test_accuracy
    return error

def print_ensemble_results(epoch, test_size, corrects, correct):
    '''Just a function to do some printing'''
    print('Validation set, Epoch {}:'.format(epoch))
    for index, num_correct in enumerate(corrects[:-1]):
        test_accuracy = 100. * num_correct / test_size
        print('Component {} Accuracy: {}/{} ({:.0f}%)'.format(
              index, num_correct, test_size, test_accuracy))
    mean_accuracy = 100. * corrects[-1] / test_size
    ensemble_accuracy = 100. * correct / test_size
    print('Posterior Mean Accuracy: {}/{} ({:.0f}%)'.format(
           corrects[-1], test_size, mean_accuracy))
    print('Ensemble Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, test_size, ensemble_accuracy))
  
def test_ensemble(model, device, test_loader, epoch, test_batch_size, num_classes, n_test_samples):
    '''Similar to test() but takes n_test_samples of samples from the
       implicit "ensemble" of networks that we get from Bayes by Backprop
    '''
    model.eval()
    correct = 0
    corrects = np.zeros(n_test_samples+1, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # put tensors on GPU
            outputs = torch.zeros(n_test_samples+1, test_batch_size,
                                  num_classes).to(device)
            for i in range(n_test_samples):
                outputs[i] = model(data, take_sample=True)
            outputs[n_test_samples] = model(data, take_sample=False) # compute output with mean weights too
            mean_output = outputs.mean(dim=0) # take mean across test samples
            preds = outputs.argmax(dim=2, keepdim=True)
            mean_pred = mean_output.argmax(dim=1, keepdim=True)
            corrects += preds.eq(target.view_as(mean_pred)).sum(dim=1).squeeze().cpu().numpy() # note 7
            correct += mean_pred.eq(target.view_as(mean_pred)).sum().item()
    print_ensemble_results(epoch, (test_batch_size * len(test_loader)), corrects, correct)
    ensemble_accuracy = 100. * correct / (test_batch_size * len(test_loader))
    error = 100 - ensemble_accuracy
    return error

def combine_params(gauss_prior_params, mixture_prior_params):
    combined = []
    combined.extend([[p] for p in gauss_prior_params])
    if mixture_prior_params:
      combined.extend(itertools.product(mixture_prior_params.nl_sigma1s,
                                        mixture_prior_params.nl_sigma2s,
                                        mixture_prior_params.pis))
    return combined

def write_results(outf, valid_error, params, bayes_params, errors):
    ModelClass, units_per_layer, optimizer_settings = params
    OptimizerClass, optimizer_kwargs = optimizer_settings
    if bayes_params:
        n_train_samples, KL_reweight, prior_form = bayes_params
        outf.write('{:.2f},{},{},{},{},{},{},{},{}\n'.format(
                  valid_error, ModelClass.__name__, units_per_layer,
                  OptimizerClass.__name__, optimizer_kwargs,
                  n_train_samples, KL_reweight, prior_form, errors))
    else:
        outf.write(('{:.2f},{},{},{},{},{},{},{},{}\n'.format(
                  valid_error, ModelClass.__name__, units_per_layer,
                  OptimizerClass.__name__, optimizer_kwargs,
                  '', '', '', errors)))

def run_experiment(args, params, outf, train_loader, valid_loader, device,
                   experiment_count, bayes_params=()):
    '''Runs an experiment with data in loaders
       with specified args (no grid search) and params/bayes_params (grid search)
       Writes output to outf
    '''
    ModelClass, units_per_layer, optimizer_settings = params
    OptimizerClass, optimizer_kwargs = optimizer_settings
    if ModelClass == BayesianNet:
        model = ModelClass(units_per_layer, args, bayes_params).to(device)
        test_fn = test_ensemble
    else:
        model = ModelClass(units_per_layer).to(device)
        test_fn = test
        
    optimizer = OptimizerClass(model.parameters(), **optimizer_kwargs)
    errors = []

    for epoch in range(1, args.epochs + 1):
      print('\nCompeted experiments: {}'.format(experiment_count))
      print('Training with: {}, {}, {}, {}, {}'.format(ModelClass.__name__,
                                           units_per_layer, 
                                           OptimizerClass.__name__,
                                           optimizer_kwargs,
                                           bayes_params))
      train(model, args.device, train_loader, optimizer, epoch, args.log_interval)
      valid_error = test_fn(model, args.device, valid_loader, epoch, args.test_batch_size,
                            args.num_classes, args.n_test_samples)
      errors.append(round(valid_error,2))
    write_results(outf, valid_error, params, bayes_params, errors)

def main():
    # For reproducability
    torch.manual_seed(0)
    if torch.cuda.is_available():
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    
    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    print('Using device: {}'.format(device))

    # Set arguments for which no grid search
    Args = namedlist('Args', ['epochs', 'batch_size', 'valid_set_frac', 'test_batch_size',
                              'num_classes', 'log_interval', 'device', 'loader_kwargs',
                              'n_test_samples', ('num_batches_train', None)]) # last two are only for BayesianNet
    args = Args(epochs=1200,
                batch_size=128, # note 9
                valid_set_frac=9952/60000, # note 9
                test_batch_size=16, # note 10
                num_classes=10, # output dimensionality
                log_interval=100, # batches to wait before logging training status
                device=device,
                loader_kwargs=loader_kwargs,
                n_test_samples=10
               )

    # Load data
    train_loader, valid_loader = get_train_valid_loader(train_batch_size=args.batch_size,
                                 valid_batch_size=args.test_batch_size,
                                 valid_size=args.valid_set_frac, 
                                 shuffle=True,
                                 **args.loader_kwargs)

    test_loader = get_test_loader(batch_size=args.test_batch_size,
                                 shuffle=True,
                                 **args.loader_kwargs) # this script doesn't use test data currently
    args.num_batches_train = len(train_loader)

    # Open file to write results; write headings
    outf = open('out_test.csv', 'w')
    outf.write('{},{},{},{},{},{},{},{},{}\n'.format(
                'final_error (valid set)', 'ModelClass', 'units_per_layer',
                'OptimizerClass', 'optimizer_kwargs',
                'n_variational_samples (when training)', 'KL_reweight',
                'prior_form', 'all_errors'))

    # Set parameters for grid search
    units_per_layer = [400, 800, 1200]
    Optimizers = [(optim.SGD, {'lr': 1e-3})]
    # these next ones are only for BayesianNet
    n_train_samples = [10]#[1,2,5,10]
    KL_reweight = [True]#[False, True]
    gauss_prior_params = [0]#[0, 1] # these are -log(sigma) values. make empty if u don't want gaussian priors
    mixture_prior_params = namedtuple('mixture_params', 'pis nl_sigma1s nl_sigma2s')(
                           [0.5],
                           [0],#[0, 1],
                           [6]) # make empty if u don't want scale mixture priors
    prior_params = combine_params(gauss_prior_params, mixture_prior_params)

    # Run experiments
    experiment_count = 0
    
    # MLP and MLP with dropout
    for parameters in itertools.product([MLP, Dropout_MLP],
                                        units_per_layer, Optimizers):
        run_experiment(args, parameters, outf, train_loader, test_loader,
                       device, experiment_count)
        experiment_count += 1
        
    # Bayesian Net
    for parameters in itertools.product([BayesianNet], units_per_layer, Optimizers):
      for bayes_params in itertools.product(n_train_samples, KL_reweight,
                                                  prior_params):
          run_experiment(args, parameters, outf, train_loader, test_loader,
                               device, experiment_count, bayes_params)
          experiment_count += 1
    
    outf.close()

if __name__ == '__main__':
    main()
