import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNet101
from dataset import Image_Dateset
import copy
from tqdm import tqdm
import pickle

torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(args):
    DEVICE = "cuda:"+args.cuda if torch.cuda.is_available() else "cpu"

    print("Use device: ", DEVICE)

    device = torch.device(DEVICE)

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = Image_Dateset(
        args.train_label_info_file_name, transform=data_transform)

    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = ResNet101(
        feature_extract=args.feature_extract,
        num_classes=args.num_classes,
        use_pretrained=args.use_pretrained).get_model().to(device)

    best_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    train_loss_history = []
    train_acc_history = []

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0
        count = 0

        for i, (inputs, label) in tqdm(enumerate(data_loader), ascii=True, total=len(data_loader)):
            count += 1
            inputs = inputs.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(inputs)

            _, predict = torch.max(output.detach(), 1)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            training_corrects += torch.sum(predict == label.detach()).item()

        training_loss = training_loss / count

        train_loss_history.append(training_loss)

        train_acc = training_corrects / len(train_dataset)

        train_acc_history.append(train_acc)

        print(
            f'Training loss: {training_loss:.4f}\taccuracy: {train_acc:.4f}\n')

        if train_acc > best_acc:
            best_acc = train_acc
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, args.model_path)

    with open(args.info_path, 'wb') as f:
        pickle.dump({
            'train_loss_history': train_loss_history,
            'train_acc_history': train_acc_history,
        }, f)


if __name__ == "__main__":
    from config import get_argparse
    parser = get_argparse()
    args = parser.parse_args()

    train(args)
