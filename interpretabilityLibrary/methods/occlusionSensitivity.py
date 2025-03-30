from typing import Tuple, Optional, Union, List, Any
import keras

from ..core.base import Explainer, Explanation

class OcclusionSensitivity(Explainer):
    """Occlusion Sensitivity explainer."""
    
    def __init__(
        self, 
        model: keras.Model, 
        window_size: Tuple[int, int] = (8, 8),
        stride: Tuple[int, int] = (4, 4),
        occlusion_value: float = 0.0
    ):
        """Initialize the explainer."""
        super().__init__(model)
        self.window_size = window_size
        self.stride = stride
        self.occlusion_value = occlusion_value
    
    def explain(
        self, 
        inputs: Any, 
        targets: Optional[Union[int, List[int], Any]] = None,
        use_jit: bool = False,
        **kwargs
    ) -> Explanation:
        """Generate occlusion sensitivity map for the given inputs."""
        ops = keras.ops
        
        # Ensure inputs is a batch
        if len(ops.shape(inputs)) == 3:
            inputs = ops.expand_dims(inputs, axis=0)
        
        # Get predictions if targets not provided
        if targets is None:
            predictions = self.model(inputs)
            if isinstance(predictions, list):
                predictions = predictions[0]
            targets = ops.argmax(predictions, axis=1)
        
        # Handle single target case
        if isinstance(targets, (int, float)) or (
            hasattr(targets, 'shape') and len(ops.shape(targets)) == 0
        ):
            targets = ops.array([int(targets)])
        elif isinstance(targets, list):
            targets = ops.array(targets)
        
        occlusion_maps = self._compute_occlusion_map(
            inputs, targets, self.window_size, self.stride, 
            self.occlusion_value, use_jit
        )
        
        return Explanation(
            attributions=occlusion_maps,
            model=self.model,
            inputs=inputs,
            targets=targets
        )
    
    def _compute_occlusion_map(
        self, 
        inputs, 
        targets,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
        occlusion_value: float,
        use_jit: bool = False
    ):
        """Compute occlusion sensitivity maps."""
        ops = keras.ops
        batch_size, height, width, channels = ops.shape(inputs)
        window_height, window_width = window_size
        stride_height, stride_width = stride
        
        # Compute baseline prediction scores
        baseline_preds = self.model(inputs)
        if isinstance(baseline_preds, list):
            baseline_preds = baseline_preds[0]
        
        # Get scores for target classes
        baseline_scores = ops.array([
            baseline_preds[i, targets[i]] for i in range(batch_size)
        ])
        
        # Initialize attribution map
        occlusion_maps = ops.zeros((batch_size, height, width, 1))
        
        # Function to create occluded inputs and get predictions
        def occlude_and_predict(inputs, h, w):
            # Create a copy of the inputs
            occluded = ops.copy(inputs)
            
            # Create occlusion patch
            occlusion_patch = ops.ones((batch_size, window_height, window_width, channels)) * occlusion_value
            
            # Update the inputs with the occlusion patch
            occluded = ops.slice_update(
                occluded,
                [0, h, w, 0],  # Start indices as a list for JAX compatibility
                occlusion_patch
            )
            
            # Get predictions for occluded inputs
            preds = self.model(occluded)
            if isinstance(preds, list):
                preds = preds[0]
            
            # Extract target scores
            return ops.array([preds[i, targets[i]] for i in range(batch_size)])
        
        # Apply JIT if using JAX backend and requested
        if use_jit and keras.backend.backend() == 'jax':
            import jax
            occlude_and_predict = jax.jit(occlude_and_predict, static_argnums=(1, 2))
        
        # Slide window over the image
        for h in range(0, height - window_height + 1, stride_height):
            for w in range(0, width - window_width + 1, stride_width):
                # Get scores with this window occluded
                occluded_scores = occlude_and_predict(inputs, h, w)
                
                # Compute importance as score drop
                score_drops = baseline_scores - occluded_scores
                
                # Update attribution map for each batch item
                for b in range(batch_size):
                    # Create a patch of the score drop value
                    score_patch = ops.ones((window_height, window_width, 1)) * score_drops[b]
                    
                    # Add to existing values in the map
                    current_values = ops.slice(
                        occlusion_maps,
                        [b, h, w, 0],
                        [1, window_height, window_width, 1]
                    )
                    
                    # Update the occlusion map
                    occlusion_maps = ops.slice_update(
                        occlusion_maps,
                        [b, h, w, 0],
                        current_values + score_patch[None, ...]
                    )
        
        return occlusion_maps