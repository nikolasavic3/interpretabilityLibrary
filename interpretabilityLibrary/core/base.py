from abc import ABC, abstractmethod
import numpy as np
import keras
from typing import Optional, Union, List

class Explanation:
    """Container for explanation results."""
    
    def __init__(
        self,
        attributions: np.ndarray,
        model: keras.Model,
        inputs: np.ndarray,
        targets: Optional[np.ndarray] = None
    ):
        self.attributions = attributions
        self.model = model
        self.inputs = inputs
        self.targets = targets

class Explainer(ABC):
    """Base class for all explainers."""
    
    def __init__(self, model: keras.Model):
        self.model = model
        if not isinstance(model, keras.Model):
            raise TypeError(f"Expected keras.Model, got {type(model)}")
    
    @abstractmethod
    def explain(self, inputs: np.ndarray, targets: Optional[Union[int, List[int], np.ndarray]] = None, **kwargs) -> Explanation:
        """Generate explanations for the given inputs."""
        pass