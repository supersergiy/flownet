import dataset

DATA_FOLDER = "./chairs/FlyingChairs_release/data"
#SPLIT_FILE  = "./chairs/FlyingChairs_train_val.txt"
SAMPLE_NUMBER = 10#22872
RANDOM_SEED = 1000
TRAIN_PORTION = 0.80
VAL_PORTION   = 0.15
TEST_PORTION  = 0.05

def main():
    labels    = {format(i, "05d"): i for i in range(SAMPLE_NUMBER)}
    all_ids   = list(labels.keys())
    partition = dataset.split(all_ids)

    training_set = FlyingChairsDataset(DATA_FOLDER, partition["train"])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = FlyingChairsDataset(DATA_FOLDER, partition["val"])
    validation_set = data.DataLoader(validation_set, **params)


if __name__ == "__main__":
    train()
