import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import pickle
from test import test

# REPRODUCIBILITY
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# args = parse_args()
# CUDA_DEVICES = args.cuda_devices
# DATASET_ROOT = args.path
CUDA_DEVICES = 0
DATASET_ROOT = '../seg_train'


config = {
    "batch_size": 32,
    'lr': 0.0001
}
num_epochs = 100


def train():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # print(DATASET_ROOT)
    train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(
        dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    # print(train_set.num_classes)
    model = VGG16(num_classes=train_set.num_classes)
    model = model.cuda(CUDA_DEVICES)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(
    #     params=model.parameters(), lr=config['lr'], momentum=0.9)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config['lr']
    )

    loss_history = []
    acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            # revise loss.data[0]-->loss.item()
            training_corrects += torch.sum(preds == labels.data)
            # print(f'training_corrects: {training_corrects}')

        training_loss = training_loss / len(train_set)
        loss_history.append(training_loss)
        training_acc = training_corrects.double() / len(train_set)
        acc_history.append(training_acc)

        # print(training_acc.type())
        # print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
        print(
            f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

        if (epoch+1) % 10 == 0:

            test_acc = test(model)
            test_acc_history.append(test_acc)

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, f'self_model-{best_acc:.02f}-best_train_acc.pth')

    with open("result_history.pickle", 'wb')as f:
        pickle.dump({
            "loss_history": loss_history,
            "acc_history": acc_history,
            "test_acc_history": test_acc_history
        }, f)


if __name__ == '__main__':
    train()
