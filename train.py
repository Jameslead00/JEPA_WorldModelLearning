from functools import partial

import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelSummary
from torchvision.transforms import Compose, Resize

from dataset import get_dataloader, GymDataset
from game import register_custom_frozen_lake, generate_frozen_lake_env, CUSTOM_FROZEN_LAKE_ID
from model.jepa import CNNEncoder, Predictor, JEPA, CNNDecoder
from lightning.pytorch.loggers import TensorBoardLogger

# Set random seed for reproducibility
seed_everything(0)


def train():
    """
    Trains the JEPA (Joint Embedding Predictive Architecture) model on a custom Frozen Lake environment.
    """
    # Define image properties
    n_input_channels = 3  # Number of input channels (e.g., RGB images)
    img_size = 64  # Image resolution
    batch_size = 32 # TODO
    max_epochs = 10  # TODO


    # Initialize the dataset and dataloader
    train_dataloader = get_dataloader(
        GymDataset(
            partial(generate_frozen_lake_env, env_id=CUSTOM_FROZEN_LAKE_ID),
            initialize_f=register_custom_frozen_lake,
            transforms=Compose([Resize(img_size)])
        ),
        batch_size=batch_size,
    )

    # TODO: Instantiate the Encoder, Decoder, and Predictor
    # Define model dimensions
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


    # Set up the logger for TensorBoard visualization
    logger = TensorBoardLogger("tb_logs", name="jepa_with_vicreg")

    # Initialize the trainer with logging and model summary callback
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, callbacks=[ModelSummary(max_depth=-1)])

    # Train the model
    trainer.fit(model, train_dataloader)

    # Save the trained model checkpoint
    if model.use_vicreg_loss:
        model_save_path = "jepa_model_vicreg.ckpt" # TODO
        trainer.save_checkpoint(model_save_path)
    else:
        print("VICReg disabled => do not save checkpoint.")


# Run the training process when the script is executed
if __name__ == "__main__":
    train()
