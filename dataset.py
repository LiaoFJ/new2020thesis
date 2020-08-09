from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "photos")  # a,b 为对照组
        self.b_path = join(image_dir, "sketches")
        self.image_filenames = [x for x in listdir(self.a_path)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((125, 100), Image.BICUBIC)
        b = b.resize((125, 100), Image.BICUBIC)

        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        # w_offset = random.randint(0, max(0, 286 - 256 - 1))
        # h_offset = random.randint(0, max(0, 286 - 256 - 1))
        # a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        # b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        return a, b


    def __len__(self):
        return len(self.image_filenames)
