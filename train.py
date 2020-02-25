from utils import load_data
from AlexNet import AlexNet
from utils import load_data

import torch
import torch.nn as nn
import logging

model_path = ["models/intel-image-classification/","","_AlxeNet.pt"]

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
    i, max_acc = 0, 0
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
            if test_acc > max_acc:
                model_path[1] = "{:.4f}".format(test_acc)
                max_acc = test_acc
                torch.save(model.state_dict(), "".join(model_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,filename='logs/AlexNet.log',format="%(message)s")


    train_dir = "dataset/intel-image-classification/seg_train/seg_train"
    val_dir = "dataset/intel-image-classification/seg_test/seg_test"

    train_iterator, val_iterator = load_data.load_data_AlexNet(train_dir, val_dir, batch_size = 32, input_size=224)

    model = AlexNet.AlexNet(num_classes = 6)
    model.apply(AlexNet.weight_init)
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_acc = 0
    for epoch in range(100):
        train_loss = train(epoch, model, train_iterator,val_iterator, optimizer, criterion)
            


