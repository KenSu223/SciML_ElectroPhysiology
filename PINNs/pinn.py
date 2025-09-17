import deepxde as dde
import numpy as np

class PINN():
    
    def __init__(self, dynamics, dim, heter, inverse):
        
        ## Dynamics
        self.dynamics = dynamics
        self.dim = dim
        self.heter = heter
        self.inverse = inverse
        
        ## PDE Parameters (initialized for 1D PINN)
        self.input = 2 # network input size 
        self.num_hidden_layers = 4 # number of hidden layers for NN 
        self.hidden_layer_size = 32 # size of each hidden layers 
        self.output = 2 # network input size 
        
        ## Training Parameters
        self.num_domain = 20000 # number of training points within the domain
        self.num_boundary = 1000 # number of training boundary condition points on the geometry boundary
        self.num_test = 1000 # number of testing points within the domain
        self.MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
        self.MAX_LOSS = 4 # upper limit to the initialized loss
        self.epochs_init = 15000 # number of epochs for training initial phase
        self.epochs_main = 1000 # number of epochs for main training phase
        self.lr = 0.0005 # learning rate
        
        ## Update constants for 2D and/or heterogeneity geometry
        self.modify_2d_const()
    
    def modify_2d_const(self):
        ## Update the PINN design for 2D and/or heterogeneity geometry
        if self.dim == 2:
            self.input = 3
            self.num_hidden_layers = 5
            self.hidden_layer_size = 60
            self.num_domain = 40000
            self.num_boundary = 4000
            self.epochs_main = 150000
        if self.heter:
            self.output = 3
        if self.inverse:
            self.lr = 0.0001
    
    def define_pinn(self, geomtime, input_data, observe_train):
        
        ## Define the network
        self.net = dde.maps.FNN([self.input] + [self.hidden_layer_size] * self.num_hidden_layers + [self.output], "tanh", "Glorot uniform")
        
        ## Select relevant PDE (Dim, Heterogeneity, forward/inverse)
        if self.dim == 1:
            pde = self.dynamics.pde_1D
        elif self.dim == 2 and self.heter:
            if self.inverse and 'd' in self.inverse:
                pde = self.dynamics.pde_2D_heter
                self.net.apply_output_transform(self.dynamics.modify_inv_heter)
            else:
                pde = self.dynamics.pde_2D_heter_forward
                self.net.apply_output_transform(self.dynamics.modify_heter)
        elif self.dim == 2 and not self.heter:
            pde = self.dynamics.pde_2D     
        
        ## Define PINN model
        self.pde_data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = self.num_domain, 
                            num_boundary=self.num_boundary, 
                            anchors=observe_train,
                            num_test=self.num_test)    
        self.model = dde.Model(self.pde_data, self.net)
        self.model.compile("adam", lr=self.lr)
        return 0
        
    def stable_init(self):
        
        ## Stabalize initialization process by capping the losses
        losshistory, _ = self.model.train(epochs=1)
        initial_loss = max(losshistory.loss_train[0])
        num_init = 1
        while initial_loss>self.MAX_LOSS or np.isnan(initial_loss):
            num_init += 1
            self.model = dde.Model(self.pde_data, self.net)
            self.model.compile("adam", lr=self.lr)
            losshistory, _ = self.model.train(epochs=1)
            initial_loss = max(losshistory.loss_train[0])
            if num_init > self.MAX_MODEL_INIT:
                raise ValueError('Model initialization phase exceeded the allowed limit')
        return 0
    
    
    def train(self, out_path, params):
        
        ## Stabalize initialization process by capping the losses
        self.stable_init()
        ## Train PINN with corresponding scheme
        if self.dim ==1:
            losshistory, train_state = self.train_1_phase(out_path, params)
        elif self.dim == 2:
            losshistory, train_state = self.train_3_phase(out_path, params)
        return self.model, losshistory, train_state
        
    def train_1_phase(self, out_path, params):
        if self.inverse:
            variables_file = "variables_" + self.inverse + ".dat"
            variable = dde.callbacks.VariableValue(params, period=1000, filename=variables_file)    
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path, callbacks=[variable])
        else:
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path)
        return losshistory, train_state
    
    def train_3_phase(self, out_path, params):
        init_weights = [0,0,0,0,1]
        if self.inverse:
            variables_file = "variables_" + self.inverse + ".dat"
            variable = dde.callbacks.VariableValue(params, period=1000, filename=variables_file)    
            ## Initial phase
            self.model.compile("adam", lr=0.0005, loss_weights=init_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path, callbacks=[variable])
            ## Main phase
            self.model.compile("adam", lr=self.lr)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path, callbacks=[variable])
            ## Final phase
            self.model.compile("L-BFGS-B")
            losshistory, train_state = self.model.train(model_save_path = out_path, callbacks=[variable])
        else:
            ## Initial phase
            self.model.compile("adam", lr=0.0005, loss_weights=init_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path)
            ## Main phase
            self.model.compile("adam", lr=self.lr)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path)
            ## Final phase
            self.model.compile("L-BFGS-B")
            losshistory, train_state = self.model.train(model_save_path = out_path)
        return losshistory, train_state
        


### Code Review for BMI 500 HW 3:
"""
The provided code defines a Physics-Informed Neural Network (PINN) class using the DeepXDE library. The class is designed to handle different dynamics, dimensions (1D and 2D), heterogeneity, and inverse problems. Below are some observations and suggestions for improvement:

Prompt used: {I have a machine learning code script in Python here which defines the network architecture for a physics informed neural network, please conduct a detailed code review with the following objectives:

Check for correctness and logic, like if the architecture is implemented correctly, or any logical mistakes
Check for efficiency, like if there would be any operation that could be optimized, or any improvement that could lead to faster computation and better memory overhead.
Check for coding best practices, like if the code is appropriately modularized, readable and maintainable, and if it follows expected coding conventions.
Write a code review for this file based on these key objectives}

AI Model used: GPT-4.1 embedded in VSCode Copilot

Code Review: pinn.py
1. Correctness and Logic

- Initialization and Architecture:
The class PINN is initialized with sensible defaults for 1D, and the modify_2d_const method updates parameters for 2D and heterogeneity/inverse problems. This is a good approach for flexibility.
The network architecture is defined using DeepXDE's FNN with customizable input, hidden layers, and output sizes. The logic for selecting the PDE and output transforms based on dim, heter, and inverse is clear and appears correct.
The define_pinn method correctly sets up the PDE, network, and DeepXDE model.
The stable_init method attempts to ensure a good initialization by capping the loss, which is a practical approach for PINNs.

- Potential Issues:
In define_pinn, the output transform is applied via self.net.apply_output_transform(...). However, DeepXDE's apply_output_transform modifies the network in-place, and repeated calls may stack transforms if define_pinn is called multiple times. Consider resetting or guarding against this.
In train_3_phase, the init_weights are hardcoded as [0,0,0,0,1]. If the number of outputs changes (e.g., due to heterogeneity), this could cause a mismatch. Consider dynamically setting this based on the output size.
The train method always calls both train_1_phase and train_3_phase for 2D, but the distinction between 1D and 2D is only by dim. If a new dimension is added, this logic will need updating.
The epochs_main is set to a very high value (150000) for 2D in modify_2d_const. This may be intentional, but could lead to unnecessarily long training times.

2. Efficiency
- Model Initialization:
The stable_init method retrains the model from scratch if the initial loss is too high. This can be inefficient if the initialization repeatedly fails. Consider logging or providing feedback on why initialization fails, or using a more robust initialization strategy.
- Training Loops:
Each phase in train_3_phase recompiles the model, which is necessary for changing optimizers/loss weights, but recompiling can be expensive. This is standard in DeepXDE, but if possible, minimize recompilations.
- Memory Usage:
The number of domain and boundary points is high for 2D (40000 and 4000). This could lead to high memory usage. Consider batching or using generators if memory becomes an issue.
- Callbacks:
The use of dde.callbacks.VariableValue is good for tracking variables, but the period is hardcoded to 1000. Consider making this configurable.

3. Coding Best Practices
- Modularity and Readability:
The code is well modularized, with clear separation between initialization, model definition, and training.
Docstrings are missing for class and methods. Adding docstrings would improve maintainability and clarity.
Variable names are generally clear, but some could be more descriptive (e.g., heter could be is_heterogeneous).
Inline comments are helpful, but could be expanded for complex logic (e.g., why certain loss weights are chosen).
- Error Handling:
The stable_init method raises a ValueError if initialization fails too many times, which is good. Consider catching and logging exceptions during training as well.
- Coding Conventions:
The code follows PEP8 for the most part, but some minor improvements:
Use self.output for output size, but the comment says "network input size" (should be "output size").
Consider using snake_case for method names (e.g., modify_2d_const is fine, but PINN methods could be more consistent).
Avoid using bare numbers for file names (e.g., "variables_" + self.inverse + ".dat"); use os.path.join for file paths.
- Extensibility:
The code is somewhat rigid for only 1D and 2D. If higher dimensions are needed, more generalization would be helpful.
The logic for selecting PDEs and transforms is nested and could be refactored for clarity.

Summary of Recommendations
- Correctness
Guard against repeated application of output transforms.
Dynamically set init_weights based on output size.
Add checks or warnings for very high epoch counts.

- Efficiency
Consider more robust model initialization strategies.
Allow configuration of callback periods and training parameters.
Monitor memory usage for large datasets.

- Best Practices
Add docstrings to all methods and the class.
Use more descriptive variable names.
Refactor nested logic for PDE selection.
Use os.path.join for file paths.
Ensure comments accurately describe the code.

Overall, the code is well-structured and functional for its intended use, but could benefit from improved robustness, configurability, and documentation.
"""