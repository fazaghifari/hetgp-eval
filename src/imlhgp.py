import numpy as np
from scipy.special import gamma
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error

class IMLHGP():

    def __init__(self, model=None, model_noise=None, v=2,
                noise_sample_size=150):
        self.model = model
        self.model_noise = model_noise
        self.noise_sample_size = noise_sample_size
        self.v = v
        self.z = None

    def _correction_factor(self, v):
        sv = np.sqrt(np.pi) / ( 2**(0.5*v) * gamma((v+1)/2) )
        return sv

    def fit(self,X,y):

        ## Step 1
        # Check alpha length and X length, only useful for BayesOpt
        # In BayesOpt, the X is iteratively added, meanwhile IMLHGP add alpha in the previous training.
        # This couse mismatch in the length of alpha and X
        if type(self.model.alpha) is np.ndarray:
            alpha_len = len(self.model.alpha)
            samp_len = X.shape[0]
            if alpha_len < samp_len:
                len_diff = samp_len - alpha_len
                add_alpha = np.array([1e-10]*len_diff)
                self.model.alpha = np.concatenate((self.model.alpha, add_alpha))
            elif alpha_len > samp_len:
                len_diff = alpha_len - samp_len
                self.model.alpha = self.model.alpha[:-len_diff]
        # Fit standard homoscedastic GP on the training dataset
        self.model.fit(X, y)
        mean_pred, std_pred =  self.model.predict(X, return_std=True)
        kern_val = np.exp(self.model.kernel_.theta)
        const_kern = kern_val[0]
        # check problem dimension and get the right lengthscale
        prob_dim = X.shape[1]
        lengthscale_kern = kern_val[1:prob_dim+1]

        ## Step 2
        # Calculate regression residuals
        r = np.abs(y - mean_pred)
        z = r**self.v
        self.z = z

        ## Step 3
        # Train GP2 on x and z
        # self.model_noise.fit(X, np.log(z))
        self.model_noise.fit(X, z)
        noise_mean_pred, noise_std_pred = self.model_noise.predict(X, return_std=True)
        noise_mean_pred = (noise_mean_pred)

        ## Step 4
        # Update most likely noise levels
        noise_mean_pred[noise_mean_pred < 0] = 0  # ensure nonnegative noise level
        noise_x_dep = self._correction_factor(self.v) * noise_mean_pred

        ## Step 5 -- specific to sklearn
        # To update noise in the correlation matrix, we can't just set model.alpha = noise.
        # We need to "retrain", however, since retraining could alter the hyperparams, we fix the hyperparameters
        # except for WhiteNoiseKernel, since we found that fixing all params would result in non-positive semidefinite matrix
        self.model.alpha= noise_x_dep + 1e-7
        # self.model.kernel = ConstantKernel(const_kern, "fixed") * RBF(length_scale=lengthscale_kern, length_scale_bounds="fixed")
        self.model.fit(X, y)
    

    def predict(self, X, return_std=None):
        """
        Make a prediction for X input. Standard deviation can be separated to two types: aleatoric (inherent noise from the data)
        and epistemic (uncertainty of the model itself).


        Params:
            X:  input data for which to make a prediction
            return_std: if True, returns mean and the full std (aleatoric+epistemic)
            return_al_std: if True, returns mean and aleatoric std
            return_ep_std: if True, returns mean and epistemic std

        """
        if return_std is None:
            result = self.model.predict(X)
        else:
            assert_msg = "return_std options are ['total','epistemic','aleatoric','multi']"
            assert return_std.lower() in ['total','epistemic','aleatoric','multi'], assert_msg

            mean, std_ep = self.model.predict(X, return_std=True) 
            if return_std.lower()=="epistemic":
                # Epistemic std
                std = std_ep
            elif return_std.lower()=="aleatoric":
                # Aleatoric std
                var_al = self.model_noise.predict(X)
                var_al[var_al < 0] = 0
                std = np.sqrt(var_al)
            elif return_std.lower()=="total":
                # Full std (epistemic + aleatoric)
                var_ep=std_ep**2
                var_al = self.model_noise.predict(X)
                var_al[var_al < 0] = 0
                std=np.sqrt(var_ep+var_al)
            else:
                # Return aleatoric and epistemic separately
                var_al = self.model_noise.predict(X)
                var_al[var_al < 0] = 0
                std_al = np.sqrt(var_al)
                std = [std_al, std_ep]
            
            result = mean, std
                
            
        return result