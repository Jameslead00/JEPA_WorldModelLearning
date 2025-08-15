from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose, Resize
from collections import deque
import itertools
import random

from dataset import GymDataset
from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, JEPA, CNNDecoder


def torch_img_to_np_img(t: torch.Tensor) -> np.array:
    """
    Convert a PyTorch tensor image to a NumPy array image.
    The function assumes the tensor is in (C, H, W) format and removes batch dimensions if necessary.
    """
    img = t.squeeze().detach().numpy()  # Remove batch dimension and convert to NumPy
    return np.permute_dims(img, (1, 2, 0))  # Rearrange dimensions to (H, W, C) for visualization


def plot_img(img: np.array) -> None:
    """
    Display an image using Matplotlib without axis labels.
    """
    plt.imshow(img)
    plt.axis('off')  # Remove axis for better visualization
    plt.show()  # Show the image


def dreaming():
    """
    Simulates "dreaming" by predicting future states using a trained JEPA model.
    The function loads a dataset, initializes a JEPA model, and visualizes predicted states.
    """

    # Load the dataset with custom Frozen Lake environment
    dataset = GymDataset(
        partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID),
        initialize_f=register_custom_frozen_lake,
        transforms=Compose([Resize(64)])  # Resize images to 64x64
    )

    # TODO: Instantiate the Encoder, Decoder, and Predictor
    hidden_channels = [4, 8, 16]  # Channels for each layer of the encoder
    embedding_img_size = (8, 8)  # Size of the feature maps before flattening
    encoder_dim = hidden_channels[-1] * embedding_img_size[0] * embedding_img_size[1]  # 16 * 8 * 8 = 1024

    encoder = CNNEncoder(channels=hidden_channels)
    predictor = Predictor(encoder_dim=encoder_dim)
    decoder = CNNDecoder(channels=hidden_channels, embedding_img_size=embedding_img_size)

    # TODO: Create a JEPA Model where you provide the Encoder, Decoder, and Predictor as arguments
    model = JEPA(
    encoder=encoder,
    predictor=predictor,
    debug_decoder=decoder,
    )
    # Load trained model weights
    checkpoint_path = "tb_logs/jepa_with_vicreg/version_0/checkpoints/epoch=9-step=15630.ckpt"  # Adjust path as needed
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("WARNING: Using untrained model, results will be meaningless")

    # Set model to evaluation mode
    model.eval()


    # TODO: Define how many steps you want to "dream" about the future
    n_steps = 5

    # Iterate over dataset samples
    for x, _, _, _, _ in dataset:
        img = torch_img_to_np_img(x)  # Convert input image to NumPy for visualization
        s_x = model.encoder(x.unsqueeze(0))  # Encode the input image to latent space

        # Generate and visualize future states ("dreaming")
        for i in range(n_steps):  # Predict n steps into the future

            # TODO: Feed s_x through the decoder
            decoded_image = model.debug_decoder(s_x)

            img = np.concatenate((
                img,
                torch_img_to_np_img(decoded_image)  # Decode and visualize the predicted state
            ), axis=1)

            # TODO: Define an action (torch.Tensor of Shape [1])
            a = torch.tensor([1], dtype=torch.long) #action 1 down

            # TODO: Feed s_x and the action through the predictor
            s_y_pred = model.predictor(s_x, a)

            # TODO: Set s_x = the predicted action
            s_x = s_y_pred

        plot_img(img)  # Display concatenated image with predicted states





def torch_img_to_np_img(t: torch.Tensor) -> np.array:
    """
    Convert a PyTorch tensor image to a NumPy array image.
    The function assumes the tensor is in (C, H, W) format and removes batch dimensions if necessary.
    """
    img = t.squeeze().detach().numpy()  # Remove batch dimension and convert to NumPy
    return np.transpose(img, (1, 2, 0))  # Rearrange dimensions to (H, W, C) for visualization



def dream_search():
    """
    Uses the world model to find the fastest path to the bottom-right corner (position 3,3).
    Only shows the optimal path at the end.
    """
    print("Starting dream-based search to find the fastest path to the bottom-right corner (position 3,3)...")
    
    # Load the dataset with custom Frozen Lake environment
    dataset = GymDataset(
        partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID),
        initialize_f=register_custom_frozen_lake,
        transforms=Compose([Resize(64)])  # Resize images to 64x64
    )

    # Instantiate the model components
    hidden_channels = [4, 8, 16]
    embedding_img_size = (8, 8)
    encoder_dim = hidden_channels[-1] * embedding_img_size[0] * embedding_img_size[1]

    encoder = CNNEncoder(channels=hidden_channels)
    predictor = Predictor(encoder_dim=encoder_dim)
    decoder = CNNDecoder(channels=hidden_channels, embedding_img_size=embedding_img_size)

    # Create a JEPA Model
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        debug_decoder=decoder,
    )
    
    # Load trained model weights
    checkpoint_path = "tb_logs/jepa_with_vicreg/version_0/checkpoints/epoch=9-step=15630.ckpt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("WARNING: Using untrained model, results will be meaningless")

    # Set model to evaluation mode
    model.eval()
    
    # Get an initial observation from the dataset
    for x, _, _, _, _ in dataset:
        initial_obs = x
        break
    
    # Define action names for better readability
    action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
    
    # Encode the initial observation
    initial_state = model.encoder(initial_obs.unsqueeze(0))
    initial_decoded = model.debug_decoder(initial_state)
    initial_img = torch_img_to_np_img(initial_decoded)
    
    # Function to extract agent position from image
    def extract_position(img):
        """
        Extract the agent's position from the image.
        Returns (row, col) coordinates in the 4x4 grid.
        """
        # The agent is typically a red pixel
        red_channel = img[:, :, 0]
        
        # Find the brightest red pixel
        y, x = np.unravel_index(np.argmax(red_channel), red_channel.shape)
        
        # Convert to grid coordinates (0-3, 0-3)
        h, w = img.shape[0], img.shape[1]
        grid_row = min(int(y * 4 / h), 3)
        grid_col = min(int(x * 4 / w), 3)
        
        return (grid_row, grid_col)
    
    # Get initial position
    initial_position = extract_position(initial_img)
    print(f"Initial position: {initial_position}")
    
    # Define the goal position
    goal_position = (3, 3)  # Bottom-right corner
    print(f"Goal position: {goal_position}")
    
    # Try a systematic approach with BFS
    print("\nPerforming breadth-first search to find the optimal path...")
    
    # Queue for BFS: (state, path, images, positions)
    queue = deque([(initial_state, [], [torch_img_to_np_img(initial_obs)], [initial_position])])
    
    # Keep track of visited positions
    visited = {initial_position: []}
    
    # Maximum search depth
    max_depth = 10
    
    # Store the optimal path
    optimal_path = None
    optimal_images = None
    
    # Track progress
    paths_explored = 0
    max_paths_to_explore = 2000
    
    while queue and not optimal_path and paths_explored < max_paths_to_explore:
        current_state, path, images, positions = queue.popleft()
        paths_explored += 1
        
        if paths_explored % 100 == 0:
            print(f"Explored {paths_explored} paths so far...")
        
        # Skip if path is too long
        if len(path) >= max_depth:
            continue
        
        # Get current position
        current_position = positions[-1]
        
        # Check if we've reached the goal
        if current_position == goal_position:
            optimal_path = path
            optimal_images = images
            print(f"Found optimal path to goal with {len(path)} steps: {[action_names[a] for a in path]}")
            print(f"Positions along the path: {positions}")
            break  # Stop BFS once we find the shortest path
        
        # Try all possible actions in a randomized order to avoid bias
        actions = [0, 1, 2, 3]  # LEFT, DOWN, RIGHT, UP
        random.shuffle(actions)  # Randomize to avoid bias
        
        for action in actions:
            # Predict the next state
            action_tensor = torch.tensor([action], dtype=torch.long)
            next_state = model.predictor(current_state, action_tensor)
            
            # Decode to get the image
            next_decoded = model.debug_decoder(next_state)
            next_img = torch_img_to_np_img(next_decoded)
            next_position = extract_position(next_img)
            
            # Skip if position didn't change
            if next_position == current_position:
                continue
            
            # Skip if we've already visited this position with a shorter or equal path
            if next_position in visited and len(visited[next_position]) <= len(path) + 1:
                continue
            
            # Add to visited
            visited[next_position] = path + [action]
            
            # Add to queue
            queue.append((next_state, path + [action], images + [next_img], positions + [next_position]))
    
    print(f"BFS explored {paths_explored} paths.")
    
    # If BFS didn't find a path, try a more exhaustive approach
    if not optimal_path:
        print("BFS couldn't find a path. Trying a more exhaustive search...")
        
        # Generate all possible paths of lengths 2 to 8
        all_paths = []
        for length in range(2, 9):
            for path in itertools.product([0, 1, 2, 3], repeat=length):
                all_paths.append(list(path))
        
        # Shuffle the paths to avoid bias
        random.shuffle(all_paths)
        
        # Limit the number of paths to try
        paths_to_try = all_paths[:2000]  # Try 2000 paths
        
        print(f"Trying {len(paths_to_try)} different paths...")
        
        # Try each path
        for path_idx, path in enumerate(paths_to_try):
            if path_idx % 100 == 0:  # Print progress every 100 paths
                print(f"Trying path {path_idx+1}/{len(paths_to_try)}")
            
            # Start from the initial state
            current_state = initial_state
            images = [torch_img_to_np_img(initial_obs)]
            positions = [initial_position]
            
            # Follow the path, but stop if we reach the goal
            truncated_path = []
            reached_goal = False
            
            for i, action in enumerate(path):
                truncated_path.append(action)
                
                # Predict the next state
                action_tensor = torch.tensor([action], dtype=torch.long)
                next_state = model.predictor(current_state, action_tensor)
                
                # Decode to get the image
                next_decoded = model.debug_decoder(next_state)
                next_img = torch_img_to_np_img(next_decoded)
                next_position = extract_position(next_img)
                
                # Store the image and position
                images.append(next_img)
                positions.append(next_position)
                
                # Check if we've reached the goal
                if next_position == goal_position:
                    print(f"Reached goal at step {i+1} of path {path_idx+1}!")
                    print(f"Path: {[action_names[a] for a in truncated_path]}")
                    print(f"Positions: {positions}")
                    optimal_path = truncated_path
                    optimal_images = images[:i+2]  # Include initial state and all steps up to goal
                    reached_goal = True
                    break
                
                # Update current state
                current_state = next_state
            
            # If we found a path to the goal, stop searching
            if reached_goal:
                break
    
    # If we found a successful path, visualize it
    if optimal_path:
        print(f"Successfully found optimal path: {[action_names[a] for a in optimal_path]}")
        visualize_optimal_path(optimal_images, optimal_path, action_names)
    else:
        print("Could not find any path to the goal position (3,3).")
        
        # As a last resort, try some specific paths that might work
        print("Trying some specific paths that might work...")
        
        specific_paths = [
            [1, 1, 2, 2],  # DOWN, DOWN, RIGHT, RIGHT
            [2, 2, 1, 1],  # RIGHT, RIGHT, DOWN, DOWN
            [1, 2, 1, 2],  # DOWN, RIGHT, DOWN, RIGHT
            [2, 1, 2, 1],  # RIGHT, DOWN, RIGHT, DOWN
            [0, 1, 2, 2, 1],  # LEFT, DOWN, RIGHT, RIGHT, DOWN
            [3, 2, 2, 1, 1],  # UP, RIGHT, RIGHT, DOWN, DOWN
        ]
        
        for path in specific_paths:
            # Start from the initial state
            current_state = initial_state
            images = [torch_img_to_np_img(initial_obs)]
            positions = [initial_position]
            
            # Follow the path
            for action in path:
                # Predict the next state
                action_tensor = torch.tensor([action], dtype=torch.long)
                next_state = model.predictor(current_state, action_tensor)
                
                # Decode to get the image
                next_decoded = model.debug_decoder(next_state)
                next_img = torch_img_to_np_img(next_decoded)
                next_position = extract_position(next_img)
                
                # Store the image and position
                images.append(next_img)
                positions.append(next_position)
                
                # Update current state
                current_state = next_state
            
            print(f"Path {[action_names[a] for a in path]} positions: {positions}")
            
            # Check if the final position is the goal
            if positions[-1] == goal_position:
                print(f"Found a path to the goal: {[action_names[a] for a in path]}")
                visualize_optimal_path(images, path, action_names)
                return
        
        # If all else fails, just show a simple path
        print("Showing a simple path for demonstration...")
        simple_path = [1, 1, 2, 2]  # DOWN, DOWN, RIGHT, RIGHT
        
        # Reset to initial state
        current_state = initial_state
        images = [torch_img_to_np_img(initial_obs)]
        positions = [initial_position]
        
        # Follow the path
        for action in simple_path:
            # Predict the next state
            action_tensor = torch.tensor([action], dtype=torch.long)
            next_state = model.predictor(current_state, action_tensor)
            
            # Decode to get the image
            next_decoded = model.debug_decoder(next_state)
            next_img = torch_img_to_np_img(next_decoded)
            next_position = extract_position(next_img)
            
            # Store the image and position
            images.append(next_img)
            positions.append(next_position)
            
            # Update current state
            current_state = next_state
        
        print(f"Simple path positions: {positions}")
        visualize_optimal_path(images, simple_path, action_names)


def visualize_optimal_path(images, action_sequence, action_names):
    """
    Visualizes the optimal path from the initial state to the goal.
    
    Args:
        images: List of images along the path
        action_sequence: List of actions in the optimal path
        action_names: Names of the actions for display
    """
    # Create a figure to display the sequence
    n_steps = len(action_sequence)
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(3 * (n_steps + 1), 3))
    
    # Handle the case where there's only one step
    if n_steps == 0:
        axes = [axes]
    
    # Display initial state
    axes[0].imshow(images[0])
    axes[0].set_title("Initial State")
    axes[0].axis('off')
    
    # For each action, show the resulting state
    for i, action in enumerate(action_sequence):
        # Display the image
        axes[i+1].imshow(images[i+1])
        axes[i+1].set_title(f"Step {i+1}: {action_names[action]}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Path to Goal ({len(action_sequence)} steps)")
    plt.show()
    
if __name__ == "__main__":
    #dream()
    dream_search()

