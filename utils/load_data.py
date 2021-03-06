import os
import torch
from torchvision import transforms, datasets

def load_data_AlexNet(train_dir, val_dir, batch_size, input_size = 224):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size = (input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(size = (input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    train_datasets = datasets.ImageFolder(train_dir, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle=True, num_workers=2)

    val_datasets = datasets.ImageFolder(val_dir, val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = batch_size, shuffle = True, num_workers = 2)

    return train_dataloader, val_dataloader

        
# def delWrongfiles(PATH):
#     import os
#     for path, dir_list, file_list in os.walk(PATH):
#         for file_name in file_list:
#             img_path = os.path.join(path, file_name)
#             print(img_path)
#             if file_name.startswith('.'):
#                 print(img_path)
#                 # os.remove(img_path)


if __name__ == "__main__":
    PATH = "/home/ying/repos/CNN_Pytorch_Implementation/"
    train_dir = PATH+"dataset/intel-image-classification/seg_train/seg_train"
    val_dir = PATH+"dataset/intel-image-classification/seg_test/seg_test"
    train_dataloader, val_dataloader = load_data_AlexNet(train_dir, val_dir, 32)
    for img, label in train_dataloader:
        print(label)

