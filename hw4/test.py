import pickle
from tqdm import tqdm
import numpy as np
import draw
from dataset import StockDataset
from model import Model

import torch
from torch.utils.data import DataLoader

device = torch.device('cuda:0')
model_path = './best.model'
test_data_path = './data/test.pickle'


def compute_difference(predict, y):
    diff = predict - y
    return np.sum(diff**2)


def test(model, test_dataloader):
    model.eval()
    result = []

    with torch.no_grad():
        for x, y in tqdm(test_dataloader, ascii=True,
                         total=len(test_dataloader)):
            x = x.to(device)
            out = model(x)

            result.append(out.cpu().detach().numpy())

    return np.array(result)


if __name__ == "__main__":
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    test_dataset = StockDataset(test_data)
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1
    )

    model = torch.load(model_path).to(device)

    result = test(model, test_dataloader)

    y = test_data[1]
    print(compute_difference(result, y))

    draw.draw_result(result, y)
