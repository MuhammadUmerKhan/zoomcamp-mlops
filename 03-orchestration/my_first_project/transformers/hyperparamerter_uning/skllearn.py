from typing import Callable, Dict, Tuple, Union
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from datetime import datetime, timezone  

from my_first_project.utils.models.sklearn import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def hyperparameter_tuning(
    training_set: Union[Dict[str, Union[Series, csr_matrix]], str],  # Allow str for debugging
    model_class_name: str,
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Callable[..., BaseEstimator],
]:
    """
    Performs hyperparameter tuning using the given training set and model class.
    
    Args:
        training_set (dict): Contains training and validation data.
        model_class_name (str): Name of the model class.
    
    Returns:
        Tuple containing best hyperparameters, feature matrix, labels, and model class details.
    """

    # ✅ Debugging: Print type of training_set
    print(f"Type of training_set: {type(training_set)}")
    print(f"Value of training_set: {training_set}")  # To check the actual content

    # ✅ If training_set is a string, load the model class instead of treating it as data
    if isinstance(training_set, str):
        model_class_name = training_set  # Assign the correct class name
        training_set = {}  # Initialize an empty dictionary to avoid further errors

    # ✅ Check if training_set is a dictionary
    if not isinstance(training_set, dict):
        raise TypeError(f"Error: Expected `training_set` to be a dictionary, but got {type(training_set)} with value: {training_set}")

    # ✅ Debugging: Print available keys
    print(f"Available keys in training_set: {list(training_set.keys())}")

    # ✅ Check if 'build' exists in training_set
    if 'build' not in training_set:
        raise ValueError("Error: Missing 'build' key in training_set. Please check your data pipeline.")

    data = training_set['build']

    # ✅ Debugging: Print type of 'build'
    print(f"'build' type: {type(data)}")

    # ✅ Ensure 'build' is a dictionary or list
    if not isinstance(data, (list, dict)):
        raise TypeError(f"Error: 'build' must be a dictionary or list, but got {type(data)}")

    # ✅ Unpack safely
    try:
        X, X_train, X_val, y, y_train, y_val, _ = data
    except ValueError as e:
        raise ValueError(f"Error unpacking 'build' data: {e}")

    # ✅ Load model class dynamically
    model_class = load_class(model_class_name)

    # ✅ Tune hyperparameters
    hyperparameters = tune_hyperparameters(
        model_class,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_evaluations=kwargs.get('max_evaluations', 50),  
        random_state=kwargs.get('random_state', 42),  
    )

    # ✅ Corrected timestamp handling
    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

    # ✅ Return results
    return hyperparameters, X, y, dict(cls=model_class, name=model_class_name)