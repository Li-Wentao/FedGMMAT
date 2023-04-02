from utils import FedGLM
import pandas as pd
import torch
import torch
from torch.autograd import grad
import numpy as np
from scipy.stats import chi2

class FedGLMMkin:
    def __init__(self, x, y, kins):
        # Get the null model from FedGLM
        self.null_model = FedGLM(x, y).fit()
        
        # Define the data
        self.X = self.null_model.x
        self.y = torch.concat(self.null_model.y)
        self.kins = kins
        
        # Init the parameters
        self.model_class = "fedglmmkin"
        self.alpha = self.null_model.beta * 1
        group_id = torch.ones(len(kins))
        group_unique = torch.unique(group_id)
        self.group_idx = [(group_id == group_unique[i]).nonzero(as_tuple=True)[0].tolist() for i in range(len(group_unique))]
        self.tau = torch.zeros(1+len(self.group_idx))
        self.fixtau = torch.zeros(1+len(self.group_idx))
        self.n = len(self.y)
        self.weights = torch.ones(self.n,1)
        self.eta = torch.tensor(np.concatenate(self.null_model.linear_predictors), requires_grad=True)
        self.mu = torch.tensor(np.concatenate(self.null_model.fitted_values))
        ans = sum(1/(1+torch.exp(-self.eta)))
        ans.backward()
        self.mu_eta = self.eta.grad
        self.Y = self.eta + (self.y - self.mu)/self.mu_eta
        self.sqrtW = self.mu_eta/torch.sqrt(1/self.weights*(self.mu*(1-self.mu)))
        self.alpha = self.null_model.beta.detach().T
        self.q = 1 # num of kinship matrix
        self.ng = len(self.group_idx)
        self.scale_residules = None
        self.converge = False

        # Init parameter adjustment
        self.tau[0] = 1
        self.fixtau[0] = 1
        self.idxtau = self.fixtau == 0


    def fit(self, max_itr=100, tol = 1e-5):
        # Model training params        
        self.max_itr = max_itr
        self.tol = 1e-5
        
        # Init fGMMAT model        
        q2 = sum(self.fixtau == 0)
        if (q2 > 0):
            self.tau[self.idxtau] = (torch.var(self.Y)/(self.q+self.ng)).repeat(q2)
            # fitglmm #
            p = self.X[0].shape[1]
            nx = [i.shape[0] for i in self.X]
            # calculate diagSigma
            diagSigma = torch.zeros(self.n, 1)
            for i in range(self.ng):
                diagSigma[self.group_idx[i]] = self.tau[i]/(self.sqrtW[self.group_idx[i]]**2)
            diagSigma = torch.flatten(diagSigma)
            # calculate Sigma
            Sigma = torch.diag_embed(diagSigma)
            self.tau[self.ng:] = self.tau[self.ng:]/torch.mean(torch.diag(torch.tensor(self.kins.values)))
            Sigma = Sigma + self.tau[self.ng:] * torch.tensor(self.kins.values)
            Sigma_i = torch.cholesky_inverse(torch.linalg.cholesky(Sigma))
            # calculate Sigma_iX
            start = 0; end = 0; Sigma_iX = 0
            for i in range(len(self.X)):
                end += nx[i]
                Sigma_iX += torch.mm(Sigma_i[start:end].T, self.X[i].double())
                start = end
            # calculate XSigma_iX
            start = 0; end = 0; XSigma_iX = 0
            for i in range(len(self.X)):
                end += nx[i]
                XSigma_iX += torch.mm(self.X[i].double().T, Sigma_iX[start:end])
                start = end
            # calculate cov
            cov = torch.cholesky_inverse(torch.linalg.cholesky(XSigma_iX))
            # calculate sigma_iXcov
            Sigma_iXcov = torch.mm(Sigma_iX, cov)
            # calculate P
            P = Sigma_i - torch.mm(Sigma_iX, Sigma_iXcov.T)
            # calculate PY
            PY = torch.mm(P, self.Y.double())
            # calculate APY
            APY = torch.mm(torch.tensor(self.kins.values), PY)
            # calculate PAPY
            PAPY = torch.mm(P, APY)
            # calculate YPAPY
            YPAPY = sum(self.Y * PAPY)
            # calculate trace of PV
            tr_PV = sum(sum(P * torch.tensor(self.kins.values)))
            # update tau
            tau0 = self.tau.clone()
            self.tau[self.idxtau] = max(0, tau0[self.idxtau] + tau0[self.idxtau]**2 * (YPAPY - tr_PV)/self.n)


        for step in range(self.max_itr):
            print(f'\nIteration {step+1}:\n')
            alpha0 = self.alpha
            tau0 = self.tau.clone()
            ######################### fitglmm_kin_ai #########################
            # calculate diagSigma
            diagSigma = torch.zeros(self.n, 1)
            for i in range(self.ng):
                diagSigma[self.group_idx[i]] = self.tau[i]/(self.sqrtW[self.group_idx[i]]**2)
            diagSigma = torch.flatten(diagSigma)
            # calculate Sigma
            Sigma = torch.diag_embed(diagSigma)
            Sigma = Sigma + self.tau[self.ng:] * torch.tensor(self.kins.values)
            Sigma_i = torch.cholesky_inverse(torch.linalg.cholesky(Sigma))
            # calculate Sigma_iX
            start = 0; end = 0; Sigma_iX = 0
            for i in range(len(self.X)):
                end += nx[i]
                Sigma_iX += torch.mm(Sigma_i[start:end].T, self.X[i].double())
                start = end
            # calculate XSigma_iX
            start = 0; end = 0; XSigma_iX = 0
            for i in range(len(self.X)):
                end += nx[i]
                XSigma_iX += torch.mm(self.X[i].double().T, Sigma_iX[start:end])
                start = end
            # calculate cov
            cov = torch.cholesky_inverse(torch.linalg.cholesky(XSigma_iX))
            # calculate sigma_iXcov
            Sigma_iXcov = torch.mm(Sigma_iX, cov)
            # calculate alpha(coefficient of glmmkin)
            self.alpha = torch.mm(cov, torch.mm(Sigma_iX.T, self.Y.double()))
            # calculate eta
            self.eta = self.Y - diagSigma.reshape(self.n,1) * (torch.mm(Sigma_i, self.Y.double()) - torch.mm(Sigma_iX, self.alpha))
        #     print(f"eta:\n{eta[idx_inv[:10]].T[0]}")
            # calculate P
            P = Sigma_i - torch.mm(Sigma_iX, Sigma_iXcov.T)
            # calculate PY
            PY = torch.mm(P, self.Y.double())
        #     # calculate wPY
        #     wPY = PY/sqrtW**2
            # calculate APY
            APY = torch.mm(torch.tensor(self.kins.values), PY)
            # calculate PAPY
            PAPY = torch.mm(P, APY)
            # calculate YPAPY
            YPAPY = sum(self.Y * PAPY)
            # calculate trace of PV
            tr_PV = sum(sum(P * torch.tensor(self.kins.values)))
            # calculate score
            score = sum(self.Y * PAPY) - tr_PV
            # calculate AI
            AI = sum(PY * torch.mm(torch.tensor(self.kins.values), PAPY))
            # calculate Dtau
            Dtau = score/AI
        #     print(f"AI:\n{AI[0]}")
        #     print(f"score:\n{score[0]}")
            ####################################################################
            # BACK to main #
            # update tau
            self.tau[self.idxtau] = tau0[self.idxtau] + Dtau
            self.tau[(self.tau < self.tol) & (tau0 < self.tol)] = 0
            while any(self.tau < 0):
                Dtau = Dtau/2
                self.tau[self.idxtau] = tau0[self.idxtau] + Dtau
                self.tau[(self.tau < self.tol) & (tau0 < self.tol)] = 0
            self.tau[self.tau < self.tol] = 0
            print(f"Variance component estimates:\n{self.tau}")
            print(f"Fixed-effect coefficients:\n{self.alpha.T[0]}")
            # calculate mu and mu.eta
        #     print(f"after ai eta:\n{eta[idx_inv[:10]].T[0]}")
            self.eta = self.eta.clone().detach().requires_grad_(True)
            self.mu = torch.sigmoid(self.eta)
        #     print(f"mu:\n{mu[idx_inv[:10]].T[0]}")
            ans = sum(1/(1+torch.exp(-self.eta)))
            ans.backward()
            self.mu_eta = self.eta.grad
        #     print(f"mu_eta:\n{mu_eta[idx_inv[:10]].T[0]}")
            # update Y
            self.Y = self.eta + (self.y - self.mu)/self.mu_eta
            self.sqrtW = self.mu_eta/torch.sqrt(1/self.weights*(self.mu*(1-self.mu)))
            self.sqrtW = self.sqrtW.type(torch.float32)
            if (2*max(max(abs(self.alpha - alpha0)/(abs(self.alpha) + abs(alpha0) + self.tol)), 
                      max(abs(self.tau - tau0)/(abs(self.tau) + abs(tau0) + self.tol))) < self.tol):
                self.converge = True
                break;

        # END of glmmkin #
        if (self.converge):
            res = self.y - self.mu
            res_var = torch.ones(self.n,1)
            for i in range(self.ng):
                res_var[self.group_idx[i]] = self.tau[i]
            self.scale_residules = res * self.weights / res_var
            self.P = P
            return self


def FedGLMMscore(model, geno_list):
    if (model.model_class == "fedglmmkin"):
        start = 0; end = 0; SCORE = 0; pG = 0; nx = [len(i) for i in model.X]
        for i in range(len(nx)):
            end += nx[i]
            g = torch.tensor(geno_list[i].fillna(0).values, dtype=torch.float64)
            SCORE += torch.mm(g, model.scale_residules[start:end])
            pG += torch.mm(model.P[start:end].T, g.T)
            start = end

        start = 0; end = 0; VAR = 0
        for i in range(len(nx)):
            end += nx[i]
            g = torch.tensor(geno_list[i].fillna(0).values, dtype=torch.float64)
            VAR += torch.mm(g, pG[start:end])
            start = end

        VAR = torch.diag(VAR)
        PVAL = 1-chi2.cdf((SCORE.T[0]**2/VAR).detach().numpy(), df=1)
        score_df = pd.DataFrame({
                         'SCORE':SCORE.T[0].detach().numpy(),
                         'VAR':VAR.detach().numpy(),
                         'PVAL':PVAL}, index=list(geno_list[0].index))
        return score_df
    else:
        raise Exception("The input model is not of type FedGLMMkin, please check your input model.")










