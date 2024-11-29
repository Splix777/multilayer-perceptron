from typing import List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field, model_validator


# Layer Configuration Models
class InputLayerConfig(BaseModel):
    """Input layer configuration model."""
    type: Literal["input"]
    input_shape: int


class DenseLayerConfig(BaseModel):
    """Dense layer configuration model."""
    type: Literal["dense"]
    units: int = Field(
        ...,
        gt=0,
        description="Number of units in the dense layer."
    )
    activation: Literal["lrelu", "relu", "tanh", "sigmoid", "softmax"]
    kernel_initializer: Literal["he_normal", "glorot_uniform"] = (
        "glorot_uniform"
    )
    kernel_regularizer: Optional[Literal["l2", "l1"]] = None


class DropoutLayerConfig(BaseModel):
    """Dropout layer configuration model."""
    type: Literal["dropout"]
    rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dropout rate (between 0 and 1)."
    )


# Union for all supported layers
LayerConfig = Union[InputLayerConfig, DenseLayerConfig, DropoutLayerConfig]


# Optimizer Configuration
class OptimizerConfig(BaseModel):
    """Optimizer configuration model."""
    type: Literal["adam", "sgd", "rmsprop"]
    learning_rate: float = Field(
        ...,
        gt=0,
        description="Learning rate must be positive."
    )


# Main Sequential Model Configuration
class SequentialModelConfig(BaseModel):
    """Sequential model configuration model."""
    name: str = Field(
        ...,
        min_length=1,
        description="Model name must not be empty."
    )
    layers: List[LayerConfig] = Field(
        ...,
        description="List of layer configurations."
    )
    optimizer: OptimizerConfig
    loss: Literal[
        "binary_crossentropy",
        "categorical_crossentropy",
        "mean_squared_error"
    ]
    batch_size: int = Field(
        ..., 
        gt=0,
        description="Batch size must be greater than 0."
    )
    epochs: int = Field(
        ...,
        gt=0, 
        description="Number of epochs must be greater than 0."
    )

    # Does further validation after the model is constructed.
    @model_validator(mode="after")
    def validate_layer_order(cls, values) -> Any:
        """
        Validate that the first layer is an input layer
        and ensure consistent layer ordering.
        """
        if not values.layers or values.layers[0].type != "input":
            raise ValueError("The first layer must be an 'input' layer.")
        for i in range(1, len(values.layers)):
            if values.layers[i].type == "input":
                raise ValueError("Input layer must be the first layer.")
            if (
                values.layers[i - 1].type == "dropout"
                and values.layers[i].type == "dropout"
            ):
                raise ValueError("Consecutive dropout layers are not allowed.")

        return values
