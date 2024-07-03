import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import RadiusNeighborsRegressor

class RNHGP():
    """ 
    Sklearn-compatible implementation for "Most Likely Heteroscedastic Gaussian Process Regression" by 
    Kristian Kersting, Christian Plagemann, Patrick Pfaff and Wolfram Burgard
    http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf

    Params:
        model: GaussianProcessRegressor from scikit-learn to fits the mean function
        model_noise: GaussianProcessRegressor from scikit-learn to fits the heteroscedastic noise
        max_iter: maximum iteration for training the MLHGP
    """
    def __init__(self, model=None, model_noise=None, 
                 max_iter=5, radius=None):
        self.model = model
        self.model_noise = model_noise
        self.max_iter = max_iter
        self.radius = radius
        self.variance_est = None
    
    def _gaussian_kernel(self, distance):
        weights = math.e**(-distance**2 / (2*self.radius**2))
        return weights
    
    def fit(self, X, y, print_noise_rmse=False):
        """Fit the model

        Args:
            X (np.array): nxm matrix, n is the number of sample, m is the number of dimension
            y (np.array): nx1 matrix
        """

        # Get the dimension of the features
        self.dim = X.shape[1] 

        noise_x_dep = self.model.alpha * np.ones(len(X))

        for i in range(self.max_iter):

            if i == 0:
                self.model.fit(X, y)
            else:
                self.model.alpha= noise_x_dep  # Put the predicted heteroscedastic noise in alpha parameter
                self.model.fit(X, y)
            
            mean_pred, std_pred =  self.model.predict(X, return_std=True)  
            if i > 0:
                std_pred= np.sqrt(std_pred**2+np.exp(self.model_noise.predict(X)))

                
        
            variance_estimator = 0.5 * ((y - mean_pred)**2 + std_pred**2)
            self.variance_est = variance_estimator
            variance_estimator = np.log(variance_estimator)

            ## Variance estimator smoothing
            if self.radius is None:
                if self.dim == 1:
                    self.radius = np.exp(self.model.kernel_.theta[1])
                else:
                    ## NOTE: Only works for GP with RBF + WhiteNoise kernel
                    multidim_radius = np.exp(self.model.kernel_.theta[1:-1]) # Excluding WhiteNoise kernel
                    self.radius = np.mean(multidim_radius)  # Taking the mean radius because RNRegressor can only handle 1 radius.

            self.model_smoothing = RadiusNeighborsRegressor(radius= self.radius, weights=self._gaussian_kernel)
            self.model_smoothing.fit(X, variance_estimator)
            variance_estimator = self.model_smoothing.predict(X)
            
            # Fitting 2nd GP
            self.model_noise.fit(X, variance_estimator)

            noise_x_dep = np.exp(self.model_noise.predict(X))

            if print_noise_rmse:
                print(f"RMSE_noise = {np.sqrt(mean_squared_error(self.model_noise.predict(X), variance_estimator))},\n pred_noise = {self.model_noise.predict(X)}, \n target_noise = {variance_estimator}")

            # At the final iteration step we have to update the input-dependent noise in the model 
            if i == (self.max_iter-1):
                self.model.alpha= noise_x_dep
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
                std = np.sqrt(np.exp(self.model_noise.predict(X)))
            elif return_std.lower()=="total":
                # Full std (epistemic + aleatoric)
                var_ep=std_ep**2
                var_al = np.exp(self.model_noise.predict(X))
                std=np.sqrt(var_ep+var_al)
            else:
                # Return aleatoric and epistemic separately
                std_al = np.sqrt(np.exp(self.model_noise.predict(X)))
                std = [std_al, std_ep]
            
            result = mean, std
                
            
        return result