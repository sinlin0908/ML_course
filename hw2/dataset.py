from torch.utils.data import Dataset
from PIL import Image
import pickle


class Image_Dateset(Dataset):
    def __init__(self,  label_infos_file_name, transform=None):
        self.x = []
        self.y = []
        self.transform = transform

        with open(label_infos_file_name, 'rb') as f:
            self.label_infos = pickle.load(f)

            for l in self.label_infos:
                self.x.append(l['file_name'])
                self.y.append(l['class'])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
