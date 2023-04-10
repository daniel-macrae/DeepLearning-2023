from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch



# this function is for the DataLoader, it makes sure the tensors within a batch are the same dimension (don't ask how, I don't know)
def collate_fn(batch):
    return tuple(zip(*batch))


# this is a class that loads the data, according to how pytorch wants it
class playersDataset(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.root = folder_path
        self.images_folder = os.path.join(folder_path, "images")
        self.labels_folder = os.path.join(folder_path, "labels")
        
        self.image_files = os.listdir(self.images_folder)
        self.label_files = os.listdir(self.labels_folder)

        self.convert_tensor = transforms.ToTensor()

    def readLabelsFile(self, file_path, index):
        boxes = []
        labels = []
        areas = []
 
        with open(file_path) as f:
            for row in f:
                annotation = [float(x) for x in row.split()]  
                #print(annotation)
                labels.append(int(annotation[0])+1)  # add 1, so that the background class is 0
                [x0, y0, x1, y1] = annotation[1:5]
                boxes.append([x0, y0, x1, y1])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)         
        labels = torch.as_tensor(labels, dtype=torch.int64)

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes" : boxes,
            "labels" : labels,
            "image_id" : torch.tensor(index),
            "area" : areas,
            "iscrowd" : iscrowd
            }

        return target


    # pytorch needs this, it returns a single (image, output) pair
    def __getitem__(self, index):
        # load and format the image file as a tensor
        
        imgPath = os.path.join(self.images_folder, self.image_files[index])
        img = Image.open(imgPath)

        input_img = self.convert_tensor(img)

        # load and format the corresponding labels
        labelPath = os.path.join(self.labels_folder, self.label_files[index])
        target = self.readLabelsFile(labelPath, index)
        
        return input_img, target

    # pytorch also needs the length of the dataset
    def __len__(self):
        return len(self.image_files)
