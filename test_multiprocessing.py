import torch
import torchvision.transforms as transforms
from hdf5dataset import Hdf5Dataset


BATCH_SIZE = 256
HDF5_PATH = "./dataset-bosch-32x32.hdf5"
NUM_WORKERS = 2

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_set = Hdf5Dataset(HDF5_PATH, transform=transform, is_test=False)
train_size = len(train_set)
test_set = Hdf5Dataset(HDF5_PATH, transform=transform, is_test=True)
test_size = len(test_set)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=NUM_WORKERS
                                           )

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=NUM_WORKERS
                                          )

for i, data in enumerate(train_loader, 0):
    inputs, labels = data
