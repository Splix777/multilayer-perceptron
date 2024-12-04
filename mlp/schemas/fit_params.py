from typing import List, Optional
from typing_extensions import Annotated
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from mlp.neural_net.callbacks.callback import Callback


class FitParams(BaseModel):
    """
    Parameters for fitting a neural network model.

    Attributes:
        X (pd.DataFrame): The input data for training.
        epochs (int): Number of epochs for training. Must be 1
            or greater.
        val_split (float, optional): Fraction of data to use
            for validation. Must be between 0 and 1.
        val_data (pd.DataFrame, optional): Validation data
            provided explicitly.
        callbacks (List[Callback]): A list of callback instances
            for monitoring training progress. Optional.
        batch_size (int): Number of samples per batch. Must
            be 1 or greater.
        verbose (bool): Whether to display progress information
            during training.
        batch_size_mode (BatchSizeMode): Mode for adjusting the batch
            size, either "auto" or "fixed".
        batch_size_factor (float): Scaling factor for batch size in
            "auto" mode. Must be between 1.0 and 2.0.
        min_batch_size (int): Minimum batch size allowed in "auto"
            mode. Must be between 1 and 1024.
        max_batch_size (int): Maximum batch size allowed in "auto"
            mode. Must be between 1 and 1024.
        batch_size_factor (float): Scaling factor for batch size in
            "auto" mode. Must be between 1.0 and 2.0.
    """
    X: pd.DataFrame
    epochs: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            description="Number of epochs for training."),
    ]
    val_split: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Fraction of data validation (0 < val_split < 1).",
        ),
    ] = 0
    val_data: pd.DataFrame
    callbacks: Annotated[
        Optional[List[Callback]],
        Field(
            default_factory=list,
            description="""A list of callback instances for monitoring
            training progress.""",
        ),
    ]
    batch_size: Annotated[
        int,
        Field(strict=True, ge=1, description="Number of samples per batch."),
    ] = 32
    verbose: bool = Field(
        False, description="Whether to display training progress."
    )
    batch_size_mode: Annotated[
        str,
        Field(
            description="""Mode for batch size adjustment.
            Must be 'auto' or 'fixed'.""",
        ),
    ] = "auto"
    min_batch_size: Annotated[
        int,
        Field(
            ge=1,
            le=2048,
            description="Minimum batch size allowed in 'auto' mode.",
        ),
    ] = 8
    max_batch_size: Annotated[
        int,
        Field(
            ge=1,
            le=2048,
            description="Maximum batch size allowed in 'auto' mode.",
        ),
    ] = 128
    batch_size_factor: Annotated[
        float,
        Field(ge=1.0, le=2.0, description="Scaling factor for batch size."),
    ] = 1.1

    @model_validator(mode="before")
    def validate_params(cls, values: dict) -> dict:
        """
        Validates the parameters for consistency and correctness.

        Args:
            values (dict): The dictionary of model input values.

        Raises:
            ValueError: If `min_batch_size` is greater than
                `max_batch_size` or if `batch_size_mode` is invalid.

        Returns:
            dict: The validated input dictionary.
        """
        # Validate batch_size_mode
        batch_size_mode: str = values.get("batch_size_mode", "auto")
        valid_modes: set[str] = {"auto", "fixed"}
        if batch_size_mode not in valid_modes:
            raise ValueError(
                f"Invalid batch_size_mode '{batch_size_mode}'. "
                f"Valid options are: {valid_modes}."
            )

        # Validate min_batch_size and max_batch_size
        min_batch_size = values.get("min_batch_size", 8)
        max_batch_size = values.get("max_batch_size", 2048)
        if min_batch_size > max_batch_size:
            raise ValueError(
                f"""min_batch_size ({min_batch_size}) cannot be greater than
                max_batch_size ({max_batch_size})."""
            )

        return values

    class Config:
        arbitrary_types_allowed: bool = True
