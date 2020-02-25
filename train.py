from utils import load_data
from utils import load_data

import torch
import torch.nn as nn
import logging

from AlexNet import AlexNet
from LeNet5 import LeNet5
from VGG import VGG

Max_acc = 0
train_dir = "dataset/intel-image-classification/seg_train/seg_train"
val_dir = "dataset/intel-image-classification/seg_test/seg_test"
learning_rate = 1e-4

Net = "VGG"

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def evaluate(epoch, model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            optimizer.zero_grad()
            fx = model(x)
            loss = criterion(fx, y)
            acc = calculate_accuracy(fx, y)
            epoch_loss += loss.item()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    print(f"Test: [{epoch}] >>> the accuracy is [{epoch_acc / len(iterator):.4f}]")       
    logging.info(f"Test: [{epoch}] >>> the accuracy is [{epoch_acc / len(iterator):.4f}]")           
    return epoch_acc / len(iterator)


def train(epoch, model, train_iterator, val_iterator, optimizer, criterion):
    epoch_loss, epoch_acc = 0, 0 
    i = 0
    model.train()
    for(x, y) in train_iterator:
        i += 1
        optimizer.zero_grad()
        fx = model(x)
        loss = criterion(fx, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        print(f"Train: Epoch: [{epoch}] >>> ({i}/{len(train_iterator)}), and the current loss is [{loss:.4f}]") 
        logging.info(f"Train: Epoch: [{epoch}] >>> ({i}/{len(train_iterator)}), and the current loss is [{loss:.4f}]") 
        
        if i % 40 == 0:
            test_acc = evaluate(epoch, model, val_iterator, optimizer, criterion)
            if test_acc > Max_acc:
                model_path[1] = "{:.4f}".format(test_acc)
                Max_acc = test_acc
                torch.save(model.state_dict(), "".join(model_path))
        # VGG
        if Net == "VGG" and epoch>0 and epoch%20 == 0:
            lr = learning_rate * (0.5 ** (epoch // 30))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


if __name__ == "__main__":
    if Net == "AlexNet":
        model_path = ["models/intel-image-classification/","","_AlxeNet.pt"]
        logging.basicConfig(level=logging.INFO,filename='logs/AlexNet.log',format="%(message)s")
        train_iterator, val_iterator = load_data.load_data_AlexNet(train_dir, val_dir, batch_size = 32, input_size=224)
        model = AlexNet.AlexNet(num_classes = 6)
        model.apply(AlexNet.weight_init)

    elif Net == "LeNet5":
        model_path = ["models/intel-image-classification/","","_LeNet5.pt"]
        logging.basicConfig(level=logging.INFO,filename='logs/LeNet5.log',format="%(message)s")
        train_iterator, val_iterator = load_data.load_data_AlexNet(train_dir, val_dir, batch_size = 32, input_size=28)
        model = LeNet5.LeNet_5(num_classes = 6)

    elif Net == "VGG":
        model_path = ["models/intel-image-classification/","","_VGG16.pt"]
        logging.basicConfig(level=logging.INFO,filename='logs/VGG16.log',format="%(message)s")
        train_iterator, val_iterator = load_data.load_data_AlexNet(train_dir, val_dir, batch_size = 32, input_size=224)
        model = VGG.VGGNet(net_arch = VGG.net_16, num_classes = 6)
        model.apply(VGG.weight_init)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(100):
        train_loss = train(epoch, model, train_iterator,val_iterator, optimizer, criterion)
            


