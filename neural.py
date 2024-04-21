from torch import nn
import copy


class MarioNet(nn.Module):
    """
    Defines a mini convolutional neural network structure specifically for inputs with a dimension of (channels, 84, 84).
    The network architecture follows this pattern:
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize the MarioNet class with specific input and output dimensions.

        Parameters:
        - input_dim (tuple): The dimensions of the input image (channels, height, width).
        - output_dim (int): The size of the output layer, corresponding to the number of classes or actions.
        """
        super().__init__()
        c, h, w = input_dim  # Unpacking the dimensions of the input.

        # Check if the input height and width are 84, if not, raise a ValueError.
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # Define the online model architecture using sequential layers.
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),  # Activation layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),  # Activation layer
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),  # Activation layer
            nn.Flatten(),  # Flatten the output for the dense layers
            nn.Linear(3136, 512),  # First dense layer
            nn.ReLU(),  # Activation layer
            nn.Linear(512, output_dim)  # Output layer
        )

        # Create a deep copy of the online model for the target model.
        self.target = copy.deepcopy(self.online)

        # Freeze the parameters of the target model; these will not be updated during training.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        Defines the forward pass of the network with an option to select between the 'online' or 'target' model.

        Parameters:
        - input (Tensor): The input data.
        - model (str): A string indicating whether to use the 'online' or 'target' model.

        Returns:
        - Tensor: The output from the selected model.
        """
        if model == 'online':
            return self.online(input)  # Return the output of the online model
        elif model == 'target':
            return self.target(input)  # Return the output of the target model
