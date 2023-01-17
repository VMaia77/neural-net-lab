import time
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import ParameterGrid
from src.model.neural_network import NeuralNetwork


class GridSearch:
    """Grid search class for hyperparameter tunning.
    """

    def __init__(self, training_data: tuple[np.ndarray, np.ndarray], modelclass: NeuralNetwork,
        start_params: dict, parameters: dict, validation_data: Tuple[np.ndarray, np.ndarray] = None) -> None:
        """
        Args:
            training_data (np.ndarray): Tuple of (X_train, y_train) training data arrays; each 2D array containing data with `float` or `integer` type.

            modelclass (NeuralNetwork): NeuralNetwork class having `set_params`, `fit`, `evaluate`, `get_params` methods.
            
            start_params (dict): Dictionary containing the start parameters of the model, 
            which will be used to initialize each instance of the model class, and therefore, these parameters will be used in all model fits.

            parameters (dict): Parameters to tune. Each key of the dictionary is a parameter and 
            the values are lists of the candidate parameters.
            
            validation_data (Tuple[np.ndarray, np.ndarray], optional): Tuple containing validation data X and y arrays. Defaults to None. 
            If None the training_data is used as validation data.
        """
        self.training_data = training_data
        self.modelclass = modelclass
        self.start_params = start_params
        self.parameters = parameters
        self.parameters_grid = ParameterGrid(parameters)
        self.validation_data = validation_data if validation_data is not None else training_data
        self.results_list = []
        self.results_df = None
        self.best_parameters = None

    def search(self) -> None:
        """Search the best (or faster) combination of hyperparameters.
        """
        for params in self.parameters_grid:
            model = self.modelclass(**self.start_params)
            model.set_params(**params)
            time_start = time.time()
            model.fit(*self.training_data)
            runtime = time.time() - time_start
            validation_loss, validation_accuracy = model.evaluate(*self.validation_data)
            results = model.get_params()
            results['loss_value'] = validation_loss
            results['accuracy'] = validation_accuracy
            results['runtime'] = runtime
            self.results_list += results,
        
        self.results_df = pd.DataFrame.from_dict(self.results_list).sort_values(by='loss', ascending=False).reset_index(drop=True)

    def get_best_params(self, by: str = 'loss') -> dict:
        """Return the best combination of hyperparameters 

        Returns:
            dict: parameters of the best combination with parameter name as keys and best parameter value as value.
        """
        if self.best_parameters:
            return self.best_parameters
            
        if by == 'loss' or by == 'runtime':
            if by == 'loss':
                by = 'loss_value'
            ascending = True

        else:
            ascending = False

        best_params = self.results_df.sort_values(by=by, ascending=ascending).reset_index(drop=True).to_dict(orient='records')[0]

        keys_to_delete = [k for k in best_params.keys() if k not in self.parameters]
        for k in keys_to_delete:
            del best_params[k]
        
        self.best_parameters = best_params

        return self.best_parameters

