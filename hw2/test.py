import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNet101
from dataset import Image_Dateset
import copy
from tqdm import tqdm
import pickle

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test(args):

    DEVICE = "cuda:"+args.cuda if torch.cuda.is_available() else "cpu"
    print("Use device: ", DEVICE)
    device = torch.device(DEVICE)

    model = torch.load(args.model_path).to(device)

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    test_set = Image_Dateset(
        args.test_label_info_file_name, transform=data_transform)

    data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    model.eval()

    total_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, total=len(data_loader), ascii=True):
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)

            _, predicted = torch.max(output.detach(), 1)
            total += labels.size(0)

            total_correct += (predicted == labels).sum().item()

        print(total)
        acc = total_correct / total

        print("Accuracy: {:.4f}".format(acc))

    return acc


if __name__ == "__main__":
    from config import get_argparse
    parse = get_argparse()
    args = parse.parse_args()
    test(args)
