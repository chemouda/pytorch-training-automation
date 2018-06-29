from alexnet import AlexNet
import torch
from torchvision import transforms, datasets
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
import convnets
from capsnet import CapsuleNet, augmentation
import argparse


parser = argparse.ArgumentParser(description='ScanVid Model Training & Evaluation')
parser.add_argument('--model-path', default=None, type=str)
parser.add_argument('--model-name', default=None, type=str)

args = parser.parse_args()

model_path = args.model_path
model_name = args.model_name
num_classes = 10

def apply_gaussian_blur(img):
    ret = cv2.GaussianBlur(np.array(img), (3, 3), 0)
    # ret = ret[None]
    return np.moveaxis(ret[None], 0, -1)


TESTSET_DIR = "/home/qamaruddin/bosch-360-lenet5-clf/ARIEL/WEB_data"

if model_name == "convnetc":
    model = convnets.ConvNetC(num_classes=10)
elif model_name == "convnetd":
    model = convnets.ConvNetD(num_classes=10)
elif model_name == "convnete":
    model = convnets.ConvNetE(num_classes=2)
elif model_name == "convnetf":
    model = convnets.ConvNetF(num_classes=2)
elif model_name == "capsnet":
    model = CapsuleNet(num_classes=10)
elif model_name == "alexnet":
    model = AlexNet(num_classes=10)

if torch.cuda.is_available():
    model = model.cuda()

model.train(False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomCrop((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        # transforms.Lambda(lambda img: apply_gaussian_blur(img)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

testset = datasets.ImageFolder(root=TESTSET_DIR, transform=transform)
test_size = len(testset)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=1
                                          )
criterion = CrossEntropyLoss()

test_loss = 0
correct = 0
for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels).type(torch.LongTensor)

    if model_name == "capsnet":
        inputs = augmentation(inputs)
        ground_truth = torch.eye(num_classes).index_select(dim=0, index=labels)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    if model_name == "capsnet":
        classes, reconstructions = model(inputs)
        loss = criterion(inputs, ground_truth, classes, reconstructions)
    else:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    test_loss += loss.data[0]

    if model_name != "capsnet":
        log_outputs = F.softmax(outputs, dim=1)
    else:
        log_outputs = classes

    pred = log_outputs.data.max(1, keepdim=True)[1]
    correct += pred.eq(labels.data.view_as(pred)).sum()

print("Testing Loss: {:.4f} \t Testing Accuracy: {:.2f} \t {}/{}".format(
    test_loss / test_size,
    100 * correct / test_size,
    correct,
    test_size
)
)
