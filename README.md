# CNN_Pytorch_Implementation
 This project is contain some of the most important classification model over the last 5 years, and very friendly for beginners.

 ### Dataset: https://www.kaggle.com/puneet6060/intel-image-classification


- [x] [LeNet5](https://github.com/YYingH/CNN_Pytorch_Implementation/tree/master/LeNet5)

    LeNet is one of the earliest neural networks. It was proposed by LeCun in 1998 to solve the problem of handwriting recognition.

- [x] [AlexNet](https://github.com/YYingH/CNN_Pytorch_Implementation/tree/master/AlexNet)

    Due to the limitation of the computation ability at that time, the structure was divided into two layers of parallel computing, but it still inherited the basic structure of LeNet. However, the changes in the details have impact on the subsequent network structure.

        1. ReLu

        2. DropOut

        3. LRN

- [x] [VGG](https://github.com/YYingH/CNN_Pytorch_Implementation/tree/master/VGG)
    For Alexnet, VGG is not a big improvement. The main improvement is the use of a small convolution kernel. The structure is build a convolution layer and followed by a maxpooling make the network is deeper and wider.
    
        1. Remove LRN layer.
    
        2. With smaller convolutional kernel, 3x3. Thereforem VGG has fewer parameters than Alexnet.
    
        3. The pooling kernel becomes smaller, the pooling kernel in VGG is 2x2, the stride is 2, the Alexnet pooling kernel is 3x3, and the step size is 2.

- [x] [GoogLeNet](https://github.com/YYingH/CNN_Pytorch_Implementation/tree/master/GoogLeNet)
    
        1. Inception module
    
        2. Global Average Pooling

