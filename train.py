import torch
from torch.utils import data

from log import logger, configure_logger
import dataset
from dataset import FlyingChairsDataset

DATA_FOLDER = "./chairs/FlyingChairs_release/data"
#SPLIT_FILE  = "./chairs/FlyingChairs_train_val.txt"

SAMPLE_NUMBER = 10#22872
TRAIN_PORTION = 0.80
VAL_PORTION   = 0.15
TEST_PORTION  = 0.05


def main():
    configure_logger()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    labels    = {format(i, "05d"): i for i in range(1, SAMPLE_NUMBER + 1)}
    all_ids   = list(labels.keys())
    partition = dataset.split(all_ids)

    # Dataset loading params
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}

    logger.info("Generating training set...")
    training_set = FlyingChairsDataset(DATA_FOLDER, partition["train"])
    training_generator = data.DataLoader(training_set, **params)

    logger.info("Generating validation set...")
    validation_set = FlyingChairsDataset(DATA_FOLDER, partition["val"])
    validation_generator = data.DataLoader(validation_set, **params)

    logger.info("Starting training")

    max_epochs = 10
    for epoch in range(max_epochs):
        logger.info("Epoch {}".format(epoch))
        for img1, img2, flo in training_generator:
            # Transfer to GPU
            img1, img2, flo = img1.to(device), img2.to(device), flo.to(device)

            # Model computations
            print ("I did it i feel like i went through a portal")

        # Validation
        with torch.set_grad_enabled(False):
            for img1, img2, flo in validation_generator:
                # Transfer to GPU
                img1, img2, flo = img1.to(device), img2.to(device), flo.to(device)

                # Model computations
                print ("I'm here")

if __name__ == "__main__":
    main()
