import numpy as np
from sklearn.metrics import mean_squared_error

class MLHGP():
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
                 max_iter=5, noise_sample_size=150):
        self.model = model
        self.model_noise = model_noise
        self.max_iter = max_iter
        self.noise_sample_size = noise_sample_size
        self.variance_est = None
    
    def fit(self, X, y, print_noise_rmse=False):

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

                
            # Fit noise
            
            # # Define sample matrix t_i^j from Section 4 Kersting et al.
            # sample_matrix = np.zeros((len(y), self.noise_sample_size))

            # for j in range(0, self.noise_sample_size):
            #     ## I think `np.eye(len(std_pred))*(std_pred)` should be `np.eye(len(std_pred))*(std_pred**2)`, but Im not sure
            #     sample_matrix[:, j] = np.random.multivariate_normal(mean_pred.reshape(len(mean_pred)), np.eye(len(std_pred))*(std_pred))

            # # Estimate variance according to the formula from Section 4 Kersting et al.
            # variance_estimator = (0.5 / self.noise_sample_size) * np.sum((np.asarray(y) - sample_matrix.T) ** 2, axis=0)
            # self.variance_est = variance_estimator
            # variance_estimator = np.log(variance_estimator+10**(-10)) #np.sqrt(variance_estimator)

            ## the std_pred should be std_pred**2, but Im not sure since it doesn't work well
            variance_estimator = 0.5 * ((y - mean_pred)**2 + std_pred)
            self.variance_est = variance_estimator
            variance_estimator = np.log(variance_estimator)
            
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