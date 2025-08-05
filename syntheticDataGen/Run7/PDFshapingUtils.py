## Author: Ricardo A. Calix, Ph.D.
## Last update: March 2023
## Release as is with no warranty
## Standard Open Source License


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import math
import functorch
import scipy.stats as stats
import random
import seaborn as sns


from numpy import hstack
from numpy import vstack
from numpy import exp
from mlxtend.plotting import heatmap
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from numpy.random import normal
from scipy.stats import norm

## coefficient of determination (R**2)
from sklearn.metrics import r2_score


class PDFshapingUtils:
    
    def __init__(self):
        self.name = 'Doctor'
        self.den = self.Dentist()
        self.car = self.Cardiologist()
        self.N_EPOCHS     = 2000                            
        self.N_EPOCHS_PDF = 2000
        self.batch_size = 16    
        self.learning_rate             =  0.001      ## 0.001        ## 0.01   ## 1e-5 
        self.learning_rate_pdfcontrol  =  0.001      ## 0.001     ## 0.00001       ## 0.000001

        ## define mean and standard deviation of target Gaussian distribution (impulse function)
        self.mean_impulse = 0.
        self.std_impulse  = 0.01
        self.kde_std      = 3.0
        self.x_range_impulse_func = None 
        self.impulse_func_vector_vals = None
        self.quadratic_weights = None
        self.N_error_range =  20          ## 20  ## 10   ## error between pred and real range (-20, 20)
        self.N_error_range_step_size = 0.01
        self.sigma_func_vector_vals = None
        
        self.detected_error_range = 0
        
        self.h = 0.9                      ## 0.7    ## 0.05     ## 0.03                    ## 0.05 >
        
        self.furnace_model_name = 'Furnace'

        self.CFD_raw_data = None
        self.headers_list = None       ##self.CFD_raw_data.columns.values.tolist()

        self.use_sigma_weighting = 0.0

        ## 0.2 more jagged, 2 more smooth, analogous to selecting number of basis functions?
        self.bandwidth = 0.2
        
        self.dict_ids_to_names = {}
        self.dict_ids_to_names_reverse = {}
        
        self.list_of_selected_column_names = None
        self.CFDdata_np = None
        
        self.input_indeces = None
        self.number_input_indeces = None
        
        self.output_indeces = None
        self.number_output_indeces = None
        
        self.random_seed = 42
        
        
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        
        self.X_train_tr  = None
        self.X_test_tr   = None
        self.y_train_tr  = None
        self.y_test_tr   = None
        
        self.x_means      =  None
        self.x_deviations =  None

        self.X_train_tr_scaled = None
        self.X_test_tr_scaled  = None
        
        self.y_means      = None
        self.y_deviations = None

        self.y_train_tr_scaled = None
        self.y_test_tr_scaled  = None
        
        self.list_metric = None
        self.the_string = None
        self.plot_low_score  = 0.7
        self.plot_high_score = 1.03
        
        self.K = lambda x: 1*torch.exp(
                                         -(      ( x/self.kde_std )**2        ) * ( 1/2 ) 
                                      ) / ( 
                                            torch.sqrt( 2 * torch.tensor(math.pi) ) * self.kde_std
                                           )
        
        self.model = None
        
        #######################################
        
        self.z            = None
        
        #######################################

 
    def show(self):
        print('In outer class')
        print('Name:', self.name)
        
    def init_z_constraints(self):
        z_init = torch.rand((1, 7)) * 0.4 + 0.3           # middle of sigmoid range
        self.z = torch.nn.Parameter(torch.logit(z_init))
        
    def get_x_from_z(self, z, clamp_min, clamp_max):
        return clamp_min + (clamp_max - clamp_min) * torch.sigmoid(z)
       
    def regularize_z(self, z, strength=1e-3):
        return strength * torch.sum(z**2)
    
    def soft_box_penalty(self, x, lower, upper, strength=10.0):
        return strength * ((torch.relu(lower - x) ** 2).sum() + (torch.relu(x - upper) ** 2).sum())
    
    def get_overshoot_margin_weights4(self):
        
        ## overshoot_margin   = torch.tensor([200.0, 50.0, 20.0])  # Limits o_fta to ≤ 2100
        ## overshoot_margin   = torch.tensor([100.0, 50.0, 10.0])       # Tighter!
        ## overshoot_margin   = torch.tensor([500.0, 100.0, 50.0])    # Tolerable overshoot per output
        ## overshoot_margin   = torch.tensor([100.0, 100.0, 50.0])    # Tolerable overshoot per output

        ## constraint_weights = torch.tensor([3.0, 8.0, 4.0])            # Penalize all more
        ## constraint_weights = torch.tensor([3.0, 16.0, 8.0])  # was [3.0, 8.0, 4.0]
        ## constraint_weights = torch.tensor([1.0, 5.0, 1.0])         # Heavier weight on hot metal temp
        
        overshoot_margin   = torch.tensor([100.0, 30.0, 10.0])  # was [200, 50, 20]
        constraint_weights = torch.tensor([1.0, 3.0, 1.0])
        
        return overshoot_margin, constraint_weights
    
    
    def get_overshoot_margin_weights3(self, k_margin=1.0, use_inverse_variance=True):
        """
        Compute overshoot margin and constraint weights.
        - overshoot_margin is set to k * std deviation
        - constraint_weights are either equal or inverse variance
        """
        overshoot_margin = (self.y_deviations * k_margin).squeeze()

        if use_inverse_variance:
            constraint_weights = (1.0 / self.y_deviations).squeeze()
        else:
            constraint_weights = torch.ones_like(self.y_deviations).squeeze()

        return overshoot_margin, constraint_weights
    
    
    def get_overshoot_margin_weights7(self, k_margin=1.0, use_inverse_variance=True):
        """
        Dynamically compute:
        - overshoot_margin as k_margin * std deviation per output
        - constraint_weights as inverse variance or uniform

        Returns:
            overshoot_margin: tolerance above target for '≥' constraints
            constraint_weights: how heavily to penalize constraint violations
        """
        
        overshoot_margin = (self.y_deviations * k_margin).squeeze()
        

        if use_inverse_variance:
            constraint_weights = (1.0 / (self.y_deviations ** 2)).squeeze()
            constraint_weights = constraint_weights / constraint_weights.mean()  # Normalize for stability
        else:
            constraint_weights = torch.ones_like(self.y_deviations).squeeze()
        
        ## constraint_weights = torch.tensor([1.0, 3.0, 1.0])

        return overshoot_margin, constraint_weights


    def get_overshoot_margin_weights(self, k_margin=1.0, use_direct_stddev=True, k_weight=1.0):
        """
        - overshoot_margin: tolerance above target for '≥' constraints
        - constraint_weights: scaled proportional to std deviation, normalized
        """
        
        k_margin = random.uniform(0.75, 1.5)   ## add more randomness

        overshoot_margin = (self.y_deviations * k_margin).squeeze()

        if use_direct_stddev:
            constraint_weights = self.y_deviations.squeeze()
            constraint_weights = constraint_weights / constraint_weights.mean()
            constraint_weights = constraint_weights * k_weight
        else:
            constraint_weights = torch.ones_like(self.y_deviations).squeeze()

        return overshoot_margin, constraint_weights

    
    
    
    def get_clamps_min_max_constraints3(self):
             
        '''
        Inputs
        (2, 'i_h2i_rate'):  0 to 100
        (3, 'i_h2_temp'):   0 to 1500 
        (4, 'i_ngi_rate'):  0 to 300
        (6, 'i_pci_rate'):  0 to 300
        (11, 'i_o2_volfract'): 21 to 40
        (12, 'i_hbtemp'): 1200 to 1500
        (13, 'i_wind_rt'): 150 to 250

        other inputs:
        (2, 'i_h2i_rate'):  0
        (3, 'i_h2_temp'):   0
        (4, 'i_ngi_rate'): 0 to 200
        (6, 'i_pci_rate'): 0 to 200
        (11, 'i_o2_volfract'): 21 to 32
        (12, 'i_hbtemp'): 1300 to 1500
        (13, 'i_wind_rt'): 150 to 250
        '''
            
        clamp_min = torch.tensor([[  0,    0,   0,    0, 21, 1300, 150 ]])     ## from Ty
        clamp_max = torch.tensor([[  0,    0, 200,  200, 32, 1500, 250 ]])


        ## clamp_min = torch.tensor([[    0,      0,   0,    0, 21, 1200, 150 ]])     ## from Ty
        ## clamp_max = torch.tensor([[  100,   1500, 300,  300, 40, 1500, 250 ]])
        
        return clamp_min, clamp_max

    
    def get_clamps_min_max_constraints(self, k=2.0):
        """
        Compute data-driven input clamps based on x_means and x_deviations.
        k = number of standard deviations to include (~2.0 = ~95% coverage)
        """
        clamp_min = (self.x_means - k * self.x_deviations).clamp(min=0.0)  # No negative rates
        clamp_max = (self.x_means + k * self.x_deviations)

        return clamp_min, clamp_max



    def gen_outputs_masks_constraints(self):
        
        target_output_not_scaled   = torch.tensor([[ 1900.0, 1700.0, 100.0    ]])  
        ## target_output_not_scaled      = torch.tensor([[ 2600.0, 1800.0,  70.0    ]])
        
        constraint_mask = torch.tensor([  2,     2,      2   ])   # 2 = ≥, 0 = no constraint
        ## constraint_mask    = torch.tensor([  3,     2,      2   ])   # o_fta is now “within range”
        
        return target_output_not_scaled , constraint_mask 
    
    
    def get_upper_lower_bounds_constraints3(self, target_output_not_scaled, constraint_mask):
        
        lower_bounds_glob    = target_output_not_scaled.clone()  
        upper_bounds_glob    = target_output_not_scaled  + torch.tensor([[200.0, 1e6, 1e6]])  ## 1e6 = 1 million

        lower_bounds_glob  = ( lower_bounds_glob - self.y_means) / self.y_deviations
        upper_bounds_glob  = ( upper_bounds_glob - self.y_means) / self.y_deviations
        
        return lower_bounds_glob, upper_bounds_glob

   

    def get_upper_lower_bounds_constraints(self, target_output_not_scaled, constraint_mask):
        
        lower_bounds_glob = torch.zeros_like(target_output_not_scaled)
        upper_bounds_glob = torch.zeros_like(target_output_not_scaled)

        # Estimate data-driven ranges based on Gaussian assumption (~86% coverage)
        
        ## k = 1.5  # std multiplier       ## try the random below for tigher or less tigh ranges
        k = torch.empty(1).uniform_(0.5, 2.0).item()  # random k between 0.5 and 2.0

        range_margin = (self.y_deviations * k).squeeze()
        floor = (self.y_means - k * self.y_deviations).squeeze()
        ceil  = (self.y_means + k * self.y_deviations).squeeze()

        for i in range(target_output_not_scaled.shape[1]):
            
            target   = target_output_not_scaled[0, i]
            mask_val = constraint_mask[i]

            if mask_val == 1:  # Equality
                lower_bounds_glob[0, i] = target
                upper_bounds_glob[0, i] = target

            elif mask_val == 2:  # ≥ constraint
                lower_bounds_glob[0, i] = target
                upper_bounds_glob[0, i] = ceil[i]  # safe upper bound

            elif mask_val == 3:  # bounded range
                lower_bounds_glob[0, i] = target
                upper_bounds_glob[0, i] = target + range_margin[i]

            else:  # No constraint (optional fallback)
                lower_bounds_glob[0, i] = floor[i]
                upper_bounds_glob[0, i] = ceil[i]

        # Normalize bounds for use in loss
        lower_bounds_glob = (lower_bounds_glob - self.y_means) / self.y_deviations
        upper_bounds_glob = (upper_bounds_glob - self.y_means) / self.y_deviations
 
        return lower_bounds_glob, upper_bounds_glob


    
        
    def read_csv_file_with_pandas(self, file_name):
        self.CFD_raw_data = pd.read_csv(file_name)
        self.headers_list = self.CFD_raw_data.columns.values.tolist()
        
        
    def print_headers_list(self):
        print(self.headers_list)
        print(len(self.headers_list))
        for i, name in enumerate(self.headers_list):
            print((i, name))
            self.dict_ids_to_names[i] = name
            self.dict_ids_to_names_reverse[name] = i
            
    def check_error_range( self, max_error_detected ):
        if max_error_detected > self.N_error_range:
            self.N_error_range =  torch.tensor(   max_error_detected.clone().detach() )
        self.initializeImpulseGaussian()
        
    
    def convert_pd_data_to_numpy(self):
        self.CFDdata_np = self.CFD_raw_data.to_numpy()
        print(self.CFDdata_np)
        print(self.CFDdata_np.shape)
        
    def gen_X_y_for_selected_indeces(self,  inputs , outputs ):
        
        self.input_indeces = inputs
        self.number_input_indeces = len(self.input_indeces)
        self.output_indeces = outputs
        self.number_output_indeces = len(self.output_indeces)
        
        self.X = self.CFDdata_np[:, self.input_indeces]
        self.y = self.CFDdata_np[:, self.output_indeces]
        
        print(self.number_input_indeces)
        print(self.number_output_indeces)
    
    def split_np_data_train_test(self, selected_test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                         self.X, 
                                         self.y, 
                                         test_size=selected_test_size, 
                                         random_state=self.random_seed
        )
        print(self.X_train.shape)
        print(self.X_test.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)


    def convert_dataset_from_np_to_torch(self):
        
        ## fix data type
        self.X_train  = self.X_train.astype(np.float32)
        self.X_test   = self.X_test.astype(np.float32)
        self.y_train  = self.y_train.astype(np.float32)
        self.y_test   = self.y_test.astype(np.float32)

        self.X_train_tr  = torch.from_numpy(self.X_train)
        self.X_test_tr   = torch.from_numpy(self.X_test)
        self.y_train_tr  = torch.from_numpy(self.y_train)
        self.y_test_tr   = torch.from_numpy(self.y_test)
        
        
    def kernel_density(self, sample):
        ## The kernel K is the basis function. Gaussian in this case.
        h = self.bandwidth
        f = lambda y: torch.mean(
                    functorch.vmap(self.K)     ((y - sample)/h)
        ) / h
        return functorch.vmap(f)
    


    def sum_of_basis_func(self, errors):
        kde_model = self.kernel_density( errors  )
        return kde_model
    
    
    
    def test_torchKDE_with_fake_data(self):
        error11 = normal(loc=7, scale=3, size=300)
        error12 = normal(loc=-7, scale=3, size=100)
        errors_EXAMPLE1 = hstack((error11, error12))

        error21 = normal(loc=3, scale=4, size=300)
        error22 = normal(loc=-3, scale=4, size=100)
        errors_EXAMPLE2 = hstack((error21, error22))

        error31 = normal(loc=4, scale=3, size=300)
        error32 = normal(loc=-4, scale=3, size=100)
        errors_EXAMPLE3 = hstack((error31, error32))

        error41 = normal(loc=6, scale=2, size=300)
        error42 = normal(loc=-6, scale=2, size=100)
        errors_EXAMPLE4 = hstack((error41, error42))

        error51 = normal(loc=5, scale=4, size=300)
        error52 = normal(loc=-5, scale=4, size=100)
        errors_EXAMPLE5 = hstack((error51, error52))

        matrix_of_errors = vstack( (errors_EXAMPLE1, errors_EXAMPLE2, errors_EXAMPLE3, errors_EXAMPLE4, errors_EXAMPLE5) )

        matrix_of_errors = torch.tensor( matrix_of_errors.T )
        print(matrix_of_errors.shape)
        
        plt.hist(matrix_of_errors[:, 0], bins=50)
        plt.show()
        plt.hist(matrix_of_errors[:, 1], bins=50)
        plt.show()
        plt.hist(matrix_of_errors[:, 2], bins=50)
        plt.show()
        plt.hist(matrix_of_errors[:, 3], bins=50)
        plt.show()
        plt.hist(matrix_of_errors[:, 4], bins=50)
        plt.show()
        
        print(matrix_of_errors[:].shape)
        print(matrix_of_errors[:])
        
        basis_func_trained_list = []
        print(matrix_of_errors.shape[1])
        for i in range(  matrix_of_errors.shape[1]   ):
            print(i)
            print(   matrix_of_errors[:,i].shape     )
            basis_func_trained_list.append(    self.sum_of_basis_func( matrix_of_errors[:,i] )        )  
            
        
        list_of_pred_dists = []
        for i in range(  matrix_of_errors.shape[1]   ):
            basis_func_trained = basis_func_trained_list[i]
            list_of_pred_dists.append(   basis_func_trained(self.x_range_impulse_func)       )           
        len(basis_func_trained_list )
        
        pred_kde_gaus_probs = torch.stack(list_of_pred_dists).T
        print(pred_kde_gaus_probs.shape)
        
        plt.hist(matrix_of_errors[:, 0], bins=50, density=True)       ## see probs instead of counts with density=True
        plt.plot(self.x_range_impulse_func, pred_kde_gaus_probs[:,0])     ## reshaped to 2D
        plt.show()

        plt.hist(matrix_of_errors[:, 1], bins=50, density=True)       ## see probs instead of counts with density=True
        plt.plot(self.x_range_impulse_func, pred_kde_gaus_probs[:,1])     ## reshaped to 2D
        plt.show()

        plt.hist(matrix_of_errors[:, 2], bins=50, density=True)       ## see probs instead of counts with density=True
        plt.plot(self.x_range_impulse_func, pred_kde_gaus_probs[:,2])     ## reshaped to 2D
        plt.show()

        plt.hist(matrix_of_errors[:, 3], bins=50, density=True)       ## see probs instead of counts with density=True
        plt.plot(self.x_range_impulse_func, pred_kde_gaus_probs[:,3])     ## reshaped to 2D
        plt.show()

        plt.hist(matrix_of_errors[:, 4], bins=50, density=True)       ## see probs instead of counts with density=True
        plt.plot(self.x_range_impulse_func, pred_kde_gaus_probs[:,4])     ## reshaped to 2D
        plt.show()
        
        print(sum(pred_kde_gaus_probs[:,0]))
        print(sum(pred_kde_gaus_probs[:,1]))
        print(sum(pred_kde_gaus_probs[:,2]))
        print(sum(pred_kde_gaus_probs[:,3]))
        print(sum(pred_kde_gaus_probs[:,4]))
        
        
    def func_plot_performance(self):
    
        list_samples = [i for i in range(len(self.list_metric))]
    
        plt.figure(figsize=(13,4))
    
        plt.scatter(list_samples, self.list_metric)
    
        ## plt.xlim(-1, 1)
        plt.ylim(self.plot_low_score, self.plot_high_score )    
        
        plt.title('metric during training ' + self.the_string)
        plt.xlabel('iteration/epoch')
        plt.ylabel('R**2')
        ## plt.legend()
    
        file_name = 'images/300dpi' + self.the_string + '.png'
        plt.savefig(file_name, dpi=300)

        plt.show()
        
    def print_individual_Rsquare(self, pred_descaled, y_test_tr):
        
        vector_pred_descaled = pred_descaled.detach().numpy()
        vector_y_test_tr     = y_test_tr.numpy()

        for i in range(len(self.output_indeces)):
            print("*****")
            print("*****")
            print(
                'Testing R**2 - Output: ' + str(i) + " " + str( self.dict_ids_to_names[self.output_indeces[i]] ), 
                r2_score( vector_pred_descaled[:, i], vector_y_test_tr[:, i] ) 
            )
     
    def plot_preds_vs_reals(self, list_preds, list_reals ):
        
        for z in range(    len(self.output_indeces)     ):
            list_preds_y = [list_preds[i] for i in range(z, len(list_preds), self.number_output_indeces)]
            list_reals_y = [list_reals[i] for i in range(z, len(list_reals), self.number_output_indeces)]

            plt.figure(figsize=(13,4))
            plt.plot(list_reals_y, label= 'real', color='r' )
            plt.plot(list_preds_y, label= 'pred')

            plt.title(str(self.dict_ids_to_names[self.output_indeces[z]]) + ' real vs prediction ' + self.the_string)
            plt.xlabel('test set samples')
            plt.ylabel('output')
            plt.legend()

            output_file_name = 'images/Y' + str(z) + self.furnace_model_name + 'RealToPredicted'
            output_file_name = output_file_name + self.the_string + '.png'
            plt.savefig(output_file_name, dpi=300)
            plt.show()
            
            
            n_bins = 20
            y_pred = np.array(list_preds_y)
            y_real = np.array(list_reals_y)
            error = y_pred - y_real
            fig, ax = plt.subplots(figsize =(10, 7))
            plt.hist(error, bins=n_bins, density = True, color='r', alpha=0.5)
            sns.distplot(error, bins=n_bins, color="blue")
            plt.title( str(self.dict_ids_to_names[self.output_indeces[z]]) + ' ' )
            plt.show()
        
   
    def print_errors_kdes(self, matrix_of_errors, pred_kde_gaus_probs):
        
        matrix_of_errors    = matrix_of_errors.detach().numpy()
        pred_kde_gaus_probs = pred_kde_gaus_probs.detach().numpy()
        
        for i in range(matrix_of_errors.shape[1]):
    
            plt.hist(matrix_of_errors[:, i], bins=50, density=True)      
            plt.plot(self.x_range_impulse_func, pred_kde_gaus_probs[:,i])  
            plt.title(   str(  self.dict_ids_to_names[self.output_indeces[i]]   )     )
            plt.show()
   
        
    def train_multiple_kernels_per_output(self, matrix_of_errors):
        basis_func_trained_list = []
        for i in range(  matrix_of_errors.shape[1]   ):
            basis_func_trained_list.append(    self.sum_of_basis_func( matrix_of_errors[:,i] )    ) 
        
        list_of_pred_dists = []
        for i in range(  matrix_of_errors.shape[1]   ):
            basis_func_trained = basis_func_trained_list[i]
            list_of_pred_dists.append(   basis_func_trained( self.x_range_impulse_func )       )
    
        pred_kde_gaus_probs = torch.stack(list_of_pred_dists).T
        return pred_kde_gaus_probs         ## 4000 x 5
  
        
        
    def gen_Dataloader_train(self):
        
        self.train_ds = TensorDataset(self.X_train_tr, self.y_train_tr_scaled)
        self.train_dl = DataLoader(self.train_ds, self.batch_size, shuffle=True)


    def standardize_X_scales(self):
        epsilon = 0.0001

        self.x_means      =  self.X_train_tr.mean(0, keepdim=True)
        self.x_deviations =  self.X_train_tr.std( 0, keepdim=True) + epsilon

        self.X_train_tr_scaled = (self.X_train_tr - self.x_means) / self.x_deviations
        self.X_test_tr_scaled  = (self.X_test_tr  - self.x_means) / self.x_deviations
        
        
    def standardize_y_scales(self):
        
        epsilon = 0.0001
        
        self.y_means      = self.y_train_tr.mean(0,  keepdim=True)
        self.y_deviations = self.y_train_tr.std( 0,  keepdim=True) + epsilon

        self.y_train_tr_scaled = (self.y_train_tr - self.y_means) / self.y_deviations
        self.y_test_tr_scaled  = (self.y_test_tr  - self.y_means) / self.y_deviations
        
    
    def initializeImpulseGaussian(self): 
        
        ## 4000   ## the error is in this range 
        self.x_range_impulse_func = torch.arange(-self.N_error_range, self.N_error_range, self.N_error_range_step_size)  
        
        mu    = self.mean_impulse
        sigma = self.std_impulse  
        x    = self.x_range_impulse_func
   
        left  = 1 / (    torch.sqrt(   2 * torch.tensor(math.pi)   ) * torch.sqrt(torch.tensor(sigma) )    )
        right = torch.exp(   -(x - mu)**2 / (2 * sigma)    )
        self.impulse_func_vector_vals = left * right
        
        self.quadratic_weights = x**2
        
    def updateImpulseGaussian_with_new_standard_deviation(self, new_standard_dev): 
        
        self.std_impulse =    new_standard_dev  
        
        mu    = self.mean_impulse
        sigma = self.std_impulse  
        x     = self.x_range_impulse_func
   
        left  = 1 / (    torch.sqrt(   2 * torch.tensor(math.pi)   ) * torch.sqrt(torch.tensor(sigma) )    )
        right = torch.exp(   -(x - mu)**2 / (2 * sigma)    )
        self.impulse_func_vector_vals = left * right
        
    def initializeImpulseToOtherShapes(self): 
        
        ## 4000   ## the error is in this range 
        self.x_range_impulse_func = torch.arange(-self.N_error_range, self.N_error_range, self.N_error_range_step_size)  
        
        mu    = -12.0
        sigma = 0.5 
        x    = self.x_range_impulse_func
        left  = 1 / (    torch.sqrt(   2 * torch.tensor(math.pi)   ) * torch.sqrt(torch.tensor(sigma) )    )
        right = torch.exp(   -(x - mu)**2 / (2 * sigma)    )
        fake_impulse_1 = left * right
        
        self.impulse_func_vector_vals = fake_impulse_1 
        
        
    def initializeSigmaVector(self): 
        
        x    = self.x_range_impulse_func
        self.sigma_func_vector_vals =  x**2
        
                 
    def if_known_gaussian_kernel_density(self, the_errors): 
        ## Kernel Density Estimation for PDF approximation assuming known gaussian error distribution
        x_range = self.x_range_impulse_func.unsqueeze(0)
        the_errors = the_errors.unsqueeze(2)
        left  = 1 / (    torch.sqrt(   2 * torch.tensor(math.pi)   )  )
        right = torch.exp(   -((x_range - the_errors)/self.h)**2 / (2)    )
        vector_vals = left * right
        ## density         = torch.mean( vector_vals, 0) / self.h    
        return vector_vals
    
       
    def print_correlation_coefficients(self):
        cm = np.corrcoef( self.CFD_raw_data[self.list_of_selected_column_names].values.T)
        hm = heatmap(cm, row_names=self.list_of_selected_column_names, column_names=self.list_of_selected_column_names, 
                     figsize=(20, 10))
        plt.show()
            
            
    class Dentist:
        def __init__(self):
            self.name = 'Dr. Savita'
            self.degree = 'BDS'
 
        def display(self):
            print("Name:", self.name)
            print("Degree:", self.degree)
            
  
    class Cardiologist:
        def __init__(self):
            self.name = 'Dr. Amit'
            self.degree = 'DM'
 
        def display(self):
            print("Name:", self.name)
            print("Degree:", self.degree)
 
 
