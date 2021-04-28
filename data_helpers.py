import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as img
import os 

class dataset(Dataset):
    def __init__(self, data, path):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.RandomAffine(90),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        img_path = os.path.join(self.path, img_name)#+'.png'
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    