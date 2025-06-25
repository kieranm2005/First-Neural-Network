# This file exports custom types and interfaces used in the SNN implementation.

from typing import Any, Dict, List, Tuple

# Define a type for the network parameters
NetworkParams = Dict[str, Any]

# Define a type for the input data
InputData = List[float]

# Define a type for the output data
OutputData = List[float]

# Define a type for a single training example
TrainingExample = Tuple[InputData, OutputData]

# Define a type for a batch of training examples
TrainingBatch = List[TrainingExample]