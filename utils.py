import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from numpy.linalg import inv
import pandas as pd
from scipy.stats import norm
import time

# device = 'cuda:15'
device = 'cpu'
def Pi(x, beta):
    return np.asarray((np.exp(x @ beta) / (1 + np.exp(x @ beta))))

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.layer1 = nn.Linear(input_size, 1, bias=False)
    def forward(self, x):
        out = torch.sigmoid(self.layer1(x))
        return out

class FedGLM:
    def __init__(self, x, y):
        self.input_size = x[0].shape[1]
        var_name = []
        for i in range(self.input_size):
            var_name += ['X' + str(i+1)]
        self.var_name = var_name
        if isinstance(x[0], pd.DataFrame):
            self.var_name = x[0].columns
            self.x = [np.array(data) for data in x]
        self.x = [torch.from_numpy(i.values.astype(np.float32)).to(device) for i in x]
        self.y = [torch.from_numpy(i.values.reshape(len(i),1).astype(np.float32)).to(device) for i in y]
        self.site = len(x)
        self.model = LogisticRegression(input_size=self.input_size).to(device)
        self.criterion = nn.BCELoss()
        self.model.layer1.weight.data.fill_(0)
        self.beta = self.model.layer1.weight
        self.df = pd.DataFrame
        self.converge = False
        self.T = np.nan
        self.linear_predictors = None
    
    def fit(self):
        t0 = time.time()
        for step in range(100):
            score = 0
            hessian = 0
            all_sample = sum([len(i) for i in self.y])
            for site in range(self.site):
                output = self.model(self.x[site])
                loss = self.criterion(output, self.y[site])
                score += grad(loss, self.beta, create_graph=True)[0] * (len(self.y[site]))/all_sample
            
            hess = []
            for i in range(score.shape[1]):
                grad2 = grad(score[0][i], self.beta, retain_graph=True)[0].squeeze()
                hess.append(grad2)
            hessian += torch.stack(hess)
            
            inv_hess = torch.inverse(hessian)
            for param in self.model.layer1.parameters():
                param.data = self.beta - score @ inv_hess

            if max(abs((score @ inv_hess)[0])) < 10 ** (-6):
                self.converge = True
                break;
 
    # Returning the statistics if converged
        if self.converge:
            beta = self.beta.detach().numpy().reshape(self.input_size,1)
            x_concat = torch.cat(self.x).numpy()
            y_concat = torch.cat(self.y).numpy()
            s = []
            for site in range(self.site):
                score = Pi(self.x[site].numpy(), beta)
                s += [score * (1-score)]
            V = np.diagflat(np.concatenate(s))
            #             V = np.diagflat(Pi(x_concat, beta) * (1 - Pi(x_concat, beta)))
            SE = np.sqrt(np.diag(inv(np.transpose(x_concat) @ V @ x_concat))).reshape(self.input_size,1)
            Z = beta/SE
            P = 2 * norm.cdf(-1 * np.abs(Z))
            CI_025  = beta - 1.959964 * SE
            CI_975  = beta + 1.959964 * SE
            self.iter = step
            self.run_time = time.time()-t0
            self.fitted_values = [Pi(self.x[site].numpy(), beta) for site in range(self.site)]
            self.df = pd.DataFrame({'Coef': np.transpose(beta)[0], 'Std.Err': np.transpose(SE)[0],
                                    'z': np.transpose(Z)[0], 'P-value': np.transpose(P)[0],
                                    '[0.025': np.transpose(CI_025)[0], '0.975]': np.transpose(CI_975)[0]},
                                    index = self.var_name)
            self.linear_predictors = [self.x[site].numpy() @ beta for site in range(self.site)]
            return self
        else:
            print('=================================================\nThe federated Logistic Regression algorithm failed to converge!!\n=================================================')
