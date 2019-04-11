"""
This code is for Experiment 3: Contextual Bandits
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from namedlist import namedlist

class Greedy(nn.Module):
    def __init__(self, in_units, hidden_units, out_units, epsilon=0):
        super().__init__()
        self.in_units = in_units
        self.fc1 = nn.Linear(in_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, out_units)
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        
    def forward(self, x):
        x = x.view(-1, self.in_units)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def loss(self, input, target, device=None):
        output = self.forward(input)
        return self.loss_fn(output, target)

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
    def __init__(self, in_units, out_units, prior_form,
                 init_weights=(-0.2, 0.2, -5, -4, -0.2, 0.2, -5, -4)):
        super().__init__()
        self.in_units = in_units
        self.out_units = out_units
        wma, wmb, wra, wrb, bma, bmb, bra, brb = init_weights
        # Weights (distribution to sample from)
        self.weight_mu = nn.Parameter(torch.Tensor(out_units, in_units).uniform_(wma, wmb)) # borrowed initalisation from nitarshan
        self.weight_rho = nn.Parameter(torch.Tensor(out_units, in_units).uniform_(wra, wrb))
        self.weight = GaussianReparam(self.weight_mu, self.weight_rho) # type: out_units x in_units matrix of GaussianReparam() objects (which we can sample from or find log_probs)
        # Biases (distribution to sample from)
        self.bias_mu = nn.Parameter(torch.Tensor(out_units).uniform_(bma, bmb))
        self.bias_rho = nn.Parameter(torch.Tensor(out_units).uniform_(bra, brb))
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
        if take_sample or self.training: 
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training: # (*)
            self.log_variational_posterior = self.weight.log_prob(weight).sum() + self.bias.log_prob(bias).sum()
            self.log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return F.linear(x, weight, bias) # F.linear does not learn a bias by default, whereas nn.linear does

class BayesianNet(nn.Module):
    def __init__(self, in_units, hidden_units, out_units, args, bayes_params, epsilon=None):
        super().__init__()
        n_train_samples, KL_reweight, prior_form = bayes_params
        self.v_samples = n_train_samples
        self.batch_size = args.batch_size
        self.num_batches = args.num_batches
        self.do_KL_reweighting = KL_reweight
        self.in_units = in_units
        self.out_units = out_units
        self.layer1 = BayesianLayer(in_units, hidden_units, prior_form,
                                    (-0.5, 0.5, -2.1, -2, -0.5, 0.5, -2.1, -2) )#(-1.5, 1.5, -5, -2, -1.5, 1.5, -5, -2) ) #(-1.27, 1.25, -1, -0.1, -1, 1, -2, -1) )
        self.layer2 = BayesianLayer(hidden_units, hidden_units, prior_form,
                                    (-0.5, 0.5, -2.1, -2, -0.5, 0.5, -2.1, -2) )#(-1, 1, -5, -2, -0.1, 0.1, -5, -2) )
        self.layer3 = BayesianLayer(hidden_units, out_units, prior_form,
                                    (-0.5, 0.5, -2.1, -2, -0.5, 0.5, -2.1, -2) )#(-0.1, 0.1, -5, -2, -0.05, -0.04, -5, -2) )
        self.epsilon = epsilon
        
    def forward(self, x, take_sample=True):
        x = x.view(-1, self.in_units)
        x = F.relu(self.layer1(x, take_sample))
        x = F.relu(self.layer2(x, take_sample))
        x = self.layer3(x, take_sample)
        return x
    
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
      
    def loss(self, input, target, device, batch_idx=None):
        """Variational free energy/negative ELBO loss function, called
           f(w, theta) in the paper
           NB calling model.loss() does a forward pass, so in train() function
           we don't need to call model(input)
        """
        outputs = torch.zeros(self.v_samples, self.batch_size, self.out_units).to(device)
        outputs_x = torch.zeros(self.v_samples+1, self.batch_size, self.out_units).to(device)
        log_priors = torch.zeros(self.v_samples).to(device)
        log_variational_posteriors = torch.zeros(self.v_samples).to(device)
        for i in range(self.v_samples):
            outputs[i] = self(input)
            log_variational_posteriors[i] = self.log_variational_posterior()
            log_priors[i] = self.log_prior()
        log_variational_posterior = log_variational_posteriors.mean()
        log_prior = log_priors.mean()
        ###
        # Here I use Jeffrey's nll function
        target_expanded = target.unsqueeze(0).repeat(self.v_samples,1,1)
        outputs_x[self.v_samples] = target
        for i in range(self.v_samples):
          outputs_x[i] = outputs[i]
        sigma = outputs_x.std(0).pow(2) # maybe try using .std(dim=0, unbiased=False) ?
        sigma_expanded = sigma.unsqueeze(0).repeat(self.v_samples,1,1)
        negative_log_likelihood = (((outputs-target_expanded).pow(2)/(2*sigma_expanded)).mean(0) + 0.5*torch.log(2*math.pi*sigma)).sum()
        ###
        if self.do_KL_reweighting:
            minibatch_weight = 1. / (2**self.num_batches - 1) * (2**(self.num_batches - batch_idx))
            loss = minibatch_weight * (log_variational_posterior - log_prior) + negative_log_likelihood
        else:
            loss = 1/self.num_batches * (log_variational_posterior - log_prior) + negative_log_likelihood
        return loss

def train(model, data, target, optimizer, device, batch_idx):
    optimizer.zero_grad()
    loss = model.loss(data, target, device, batch_idx)
    loss.backward()
    optimizer.step()
    return loss.item()

def compute_reward(mushroom, eat):
  is_edible = True if mushroom[0] == 'e' else False
  if eat:
    if is_edible:
      reward = 5
    else:
      reward = np.random.choice([5, -35])
  else:
    reward = 0
  return reward

def encode_input(mushroom, eat):
  context = mushroom[1:]
  context_enc = np.array([float(ord(c)) for c in context]) # vector of ascii encoding of features... is this fine?
  action_enc = [1,0] if eat else [0,1]
  return np.concatenate([context_enc, action_enc])

def run_experiment(agent, mushrooms, optimizer, args):
    n_mushrooms, n_attribs = np.shape(mushrooms)
    # Fill the buffer with 4096 randomly chosen (context_action, true_reward) tuples
    buffer = torch.empty(size=(4096, 25, 1))
    for i in range(4096):
      mushroom = mushrooms[np.random.randint(n_mushrooms)]
      eat = np.random.choice([True, False])
      reward = compute_reward(mushroom, eat)
      context_action_enc = encode_input(mushroom, eat)
      enc = np.concatenate([context_action_enc, [reward]])
      enc = torch.unsqueeze(torch.from_numpy(enc).float(), dim=1)
      buffer[i] = enc
    
    # Play against the bandit args.num_rounds times and train for 4096 steps per interaction
    cum_regret = 0
    cum_regrets = []
    is_eps_greedy = True if agent.epsilon else False
    agent_repr = 'BBB' if isinstance(agent, BayesianNet) else 'Greedy {}'.format(agent.epsilon)
    file_name = '{}.csv'.format(agent_repr)
    outf2 = open(file_name, 'w')
    outf2.write('Step,Eat,Regret\n')
    agent.train()
    for step in range(args.num_rounds):
        # one interaction with the mushroom bandit
        mushroom = mushrooms[np.random.randint(n_mushrooms)]
        eat_mush = encode_input(mushroom, eat=True)
        not_eat_mush = encode_input(mushroom, eat=False)
        # make to tensors, unsqueeze, put on device
        eat_mush = torch.unsqueeze(torch.from_numpy(eat_mush).float(), dim=1).to(args.device)
        not_eat_mush = torch.unsqueeze(torch.from_numpy(not_eat_mush).float(), dim=1).to(args.device)
        # use NN to find expected reward of eating and not eating mushroom
        E_r_eat = agent(eat_mush)
        E_r_not_eat = agent(not_eat_mush)
        do_eat = True if E_r_eat > E_r_not_eat else False
        # possibly change this action according to epsilon
        if is_eps_greedy:
            pick_random_action = np.random.choice([True, False], p=[agent.epsilon, 1-agent.epsilon])
            if pick_random_action:
              do_eat = np.random.choice([True, False])
        reward = compute_reward(mushroom, do_eat)
        # add new (context_action, true_reward) tuple to buffer
        context_action_enc = encode_input(mushroom, do_eat)
        enc = np.concatenate([context_action_enc, [reward]])
        enc = torch.unsqueeze(torch.from_numpy(enc).float(), dim=1)
        # push and pop from queue
        buffer[:-1] = buffer[1:]
        buffer[-1] = enc
        # compute regret and add to list
        is_edible = True if mushroom[0] == 'e' else False
        oracle_reward = 5 if is_edible else 0
        regret = oracle_reward - reward
        cum_regret += regret
        cum_regrets.append(cum_regret)
        # write step results to file
        outf2.write('{},{},{}\n'.format(step, do_eat, regret))
        if step % args.log_interval == 0:
       		print('Training {}, step {}'.format(agent_repr, step))
        for batch_idx in range(64):
            minibatch = buffer[np.random.randint(4096, size=64), :]
            inputs = minibatch[:, :-1].to(args.device)
            targets = minibatch[:, -1].to(args.device) 
            loss = train(agent, inputs, targets, optimizer, args.device, batch_idx) # trains for 4096 steps
            if step % args.log_interval == 0:
            	print('Batch {} loss: {}'.format(batch_idx, loss))
    outf2.close()
    return cum_regrets

def main():
    # For reproducability
    torch.manual_seed(0)
    if torch.cuda.is_available():
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))

    # Load data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    mush_pd = pd.read_csv(url)
    mushrooms = mush_pd.values # convert pd.dataframe object to np.array

    # Define experiment parameters
    Args = namedlist('Args', ['batch_size', 'num_batches',
                              'num_rounds', 'device', ('log_interval', 1000)])
    args = Args(batch_size=64,
                num_batches=64,
                num_rounds=50000,
                device=device,
                log_interval=100)

    # greedy_agent = Greedy(24, 100, 1).to(device)
    # one_greedy_agent = Greedy(24, 100, 1, 0.01).to(device)
    # five_greedy_agent = Greedy(24, 100, 1, 0.05).to(device)

    outf = open('out.csv', 'w')
    greedy_regrets = run_experiment(greedy_agent, mushrooms,
                                    optim.Adam(greedy_agent.parameters()), args)
    outf.write('Greedy,{}\n'.format(greedy_regrets))
    one_greedy_regrets = run_experiment(one_greedy_agent, mushrooms,
                                  optim.Adam(one_greedy_agent.parameters()), args)
    outf.write('1-Greedy,{}\n'.format(one_greedy_regrets))
    five_greedy_regrets = run_experiment(five_greedy_agent, mushrooms,
                                  optim.Adam(five_greedy_agent.parameters()), args)
    outf.write('5-Greedy,{}\n'.format(five_greedy_regrets))

    # BayesianNet
    variational_samples_train = 10 # authors specify they use 2 for bandits task, but that didn't work, so now we'll try 10
    do_KL_reweight = True # but the rest of the hyperparameters are left unspecified
    nl_sigma1 = 0
    nl_sigma2 = 6
    pi = 0.5
    spike_slab_prior = [nl_sigma1, nl_sigma2, pi]
    gaussian_prior_nl_sigma = [3]
    bayes_params = [variational_samples_train,
                    do_KL_reweight, spike_slab_prior]

    BBB_agent = BayesianNet(24, 100, 1, args, bayes_params).to(device)
    BBB_regrets = run_experiment(BBB_agent, mushrooms, optim.Adam(BBB_agent.parameters(), lr=1e-4),
                                 args)

    outf.write('BBB,{}\n'.format(BBB_regrets))
    outf.close()

    plt.plot(greedy_regrets, label	='Greedy')
    plt.plot(one_greedy_regrets, label='1% Greedy')
    plt.plot(five_greedy_regrets, label='5% Greedy')
    plt.plot(BBB_regrets, label='Bayes by Backprop')
    plt.title('Cumulative regret on mushroom bandit task for various agents')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.yscale('log')
    plt.savefig('regrets.png', dpi=400, bbox_inches='tight')

if __name__ == '__main__':
    main()
