import pickle
import numpy as np
from tqdm import tqdm
from model import Model
from dataset import StockDataset
from draw import draw_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.cuda.manual_seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0')

data_path = './data/train.pickle'
epochs = 50
lr = 0.0001


def train(model, dataloader):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_loss_h = []

    for epoch in range(epochs):

        epoch_loss = 0.0
        count = 0

        print(f'Epoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        for x, y in tqdm(dataloader, total=len(dataloader), ascii=True):

            count += 1

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= count

        epoch_loss_h.append(epoch_loss)

        print(f"epoch {epoch+1} loss: {epoch_loss:.4f}")

    print("save model.....")
    torch.save(model, 'best.model')
    print("end")

    return epoch_loss_h


if __name__ == "__main__":

    with open(data_path, 'rb') as f:
        train_data = pickle.load(f)

    train_dataset = StockDataset(train_data)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=10,
        num_workers=1
    )

    model = Model(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.5
    ).to(device)

    print(model)

    print([name for name, _ in model.named_parameters()])

    loss = train(model, train_dataloader)

    print("draw loss....")
    draw_loss(np.array(loss))
