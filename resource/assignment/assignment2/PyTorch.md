# Assignment2之PyTorch实践

## 0.说在前面

本节更新week8作业的PyTorch.ipynb，顺便一起来学习一下PyTorch的一些基本用法！

下面一起来实践吧！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.准备工作

在这一部分，需要注意几个函数，分别如下：

### 1.1 transform

`T.Compose`将多个transforms的list组合起来。

```python
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
```

### 1.2 ToTensor

`transforms.ToTensor() `将 PIL.Image/numpy.ndarray 数据进转化为torch.FloadTensor，并归一化到[0, 1.0]。

- 取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；
- 形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。

```python
T.ToTensor()
```

### 1.3 Normalize

归一化对神经网络是非常重要的，那么如何归一化到[-1.0，1.0]？

这里就用到了`transforms.Normalize()`，channel=（channel-mean）/std。

上述可转化为：

```python
T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
```

### 1.4 datasets

使用datasets下载数据集。

```python
import torchvision.datasets as dset
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=transform)
```

### 1.5 DataLoader

组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。

可以把下面通俗的理解为：使用采样器随机在数据集中采样，并用DataLoader将采样器与数据集进行装载，然后返回的结果是个迭代器，取出数据得通过for循环取出！

```python
from torch.utils.data import DataLoader
from torch.utils.data import sampler
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
```

### 1.6 GPU与CPU

我觉得这一段代码写的很棒，是因为简洁明了的将cpu与gpu一同来进行判别，直接确定你最后使用的是gpu还是cpu的设备去运行！

```python
USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Constant to control how frequently we print train loss
print_every = 100
print('using device:', device)
```

## 2.Barebones PyTorch

### 2.1 Flatten Function

这一节来实现PyTorch中的Flatten函数！

这里做的工作在注释中给出了明确的说明，我们来看一下：

The flatten function below first reads in the N, C, H, and W values from a given batch of data, and then returns a "view" of that data. "View" is analogous to numpy's "reshape" method: it reshapes x's dimensions to be N x ??, where ?? is allowed to be anything (in this case, it will be C x H x W, but we don't need to specify that explicitly.

这里有几个地方非常重要，第一：view类似与numpy中reshape方法，主要是将维度变为Nx?? 而两个问号又该是什么值呢，这就引出了第二个关键点：两个问号不需要明确指定。

那么根据前面的两个关键点，就可以明白下面函数所要做的事情了，那就是将(N,C,H,W)维度变为(N,CxHxW)!不需要明确指定就是-1即可完成，而reshape用view函数，那么这样的话，下面函数就so easy了！

### 2.2 Two-Layer Network

这一节来实现两层神经网络！注释中说明，这一节不用写任何代码，但你需要理解！

 A fully-connected neural networks; the architecture is:NN is fully connected -> ReLU -> fully connected layer.

全连接网络架构为：NN->ReLU->NN

这里的x.mm解释一下：x是一个pytorch 张量，x.mm使用了pytorch里面的矩阵乘法函数，作用就是实现x与w1的矩阵相乘，是真正的矩阵相乘，而不是对应元素相乘！

这里我们来看注释维度！

```python
注意点：d1 * ... * dM = D 
Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).
```

具体实现如下：先转化x维度为(N,D)，再通过矩阵乘法(N,D)x(D,H)得到(N,H)，此时是x.mm(w1)的结果，也就完成了NN层这一步，接着就是ReLU，那么用F.relu()完成了ReLU层，最后再做一次矩阵乘法，(N,H)x(H,C)，得到(N,C)，也就是x.mm(w2)的结果，最后返回即可！

```python
import torch.nn.functional as F 
x = flatten(x)  
w1, w2 = params
x = F.relu(x.mm(w1))
x = x.mm(w2)
```
后面则是两层神经网络的测试，可以自己走一遍，这里就不阐述了！

### 2.3 Three-Layer ConvNet

为什么前面不写代码呢，就是因为让你县看懂，看懂完了，来实现三层卷积。

卷积架构如下：

1. A convolutional layer (with bias) with `channel_1` filters, each with shape `KW1 x KH1`, and zero-padding of two
2. ReLU nonlinearity
3. A convolutional layer (with bias) with `channel_2` filters, each with shape `KW2 x KH2`, and zero-padding of one
4. ReLU nonlinearity
5. Fully-connected layer with bias, producing scores for C classes.

接着来看一下给定的要求：

```python
输入:
- x: (N, 3, H, W)
- params: should contain the following:
  -conv_w1 : (channel_1, 3, KH1, KW1) 第一层卷积权重
  -conv_b1 : (channel_1,) 第一层卷积偏值
  -conv_w2 : (channel_2, channel_1, KH2, KW2) 第二层卷积权重
  -conv_b2 : (channel_2,) 第二层卷积偏值
  - fc_w: 全连接层权重
  - fc_b: 全连接层偏值

返回:
- scores: (N,C)
```

具体解释看注释！

```python
# 完成卷积一层操作(实现通过F.conv2d)
# 填入输入数据，权重，偏值，步长及零填补(看上面卷积架构)
# 后面的按照上面架构来就行了
# # shape=(N,channel_1,H,W)
conv1 = F.conv2d(x, weight=conv_w1, bias=conv_b1,stride=1, padding=2)
# shape=(N,channel_1,H,W)
relu1 = F.relu(conv1)
# shape=(N,channel_2,H,W)
conv2 = F.conv2d(relu1, weight=conv_w2, bias=conv_b2, stride=1,padding=1)
# shape=(N,channel_2,H,W)
relu2 = F.relu(conv2)
# 利用上面的两层神经网络的实现，直接调用即可！
# shape=(N,CxHxW)
relu2_flat = flatten(relu2)
# shape=(N,C)
scores = relu2_flat.mm(fc_w) + fc_b
```

### 2.4 Initialization

 让我们编写几个实用程序方法来初始化我们模型的权重矩阵。

- `random_weight(shape)`  

  用Kaiming归一化方法初始化权重张量。

- `zero_weight(shape)`

   用全零初始化权重张量。用于实例化偏差参数。

### 2.5 Check Accuracy

在训练模型时，我们将使用以下函数来检查我们的模型在训练或验证集上的准确性。

在检查精度时，我们不需要计算任何梯度;因此，当我们计算分数时，我们不需要PyTorch为我们构建计算图。

### 2.6 Training Loop

 我们现在可以建立一个基本的训练循环来训练我们的网络。我们将使用没有动量的随机梯度下降来训练模型。我们将使用torch.functional.cross_entropy来计算损失; 训练循环将神经网络函数，初始化参数列表（在我们的示例中为[w1，w2]）和学习速率作为输入。

### 2.7 Train a Two-Layer Network

现在我们准备好运行训练循环了。我们需要为完全连接的权重w1和w2明确地分配张量。

CIFAR的每个小批量都有64个例子，因此张量形状为[64,3,32,32]。

展平后，x形应为[64,3 * 32 * 32]。这将是w1的第一个维度的大小。 w1的第二个维度是隐藏层大小，它也是w2的第一个维度。

最后，网络的输出是一个10维向量，表示10个类的概率分布。您不需要调整任何超参数，但在训练一个纪元后，您应该看到精度超过40％！

### 2.8 Training a ConvNet

这里是调用了上述初始化函数，初始化w与b，由于传递的是shape，那么我们可以根据在上面的卷积神经网络注释的提示里面的shape来进行编写，上面的注释如下：

```python
-conv_w1 : (channel_1, 3, KH1, KW1) 第一层卷积权重
-conv_b1 : (channel_1,) 第一层卷积偏值
-conv_w2 : (channel_2, channel_1, KH2, KW2) 第二层卷积权重
-conv_b2 : (channel_2,) 第二层卷积偏值
```

KH1与KW1，KH2与KW2是多少呢？这个注释也给出了，看本节的注释如下：

1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2
2. ReLU
3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1
4. ReLU
5. Fully-connected layer (with bias) to compute scores for 10 classes

那么最后就直接传参就行了！

```python
conv_w1 = random_weight((channel_1,3,5,5))
conv_b1 = zero_weight((channel_1,))
conv_w2 = random_weight((channel_2,channel_1,3,3))
conv_b2 = zero_weight((channel_2,))
fc_w = random_weight((channel_2*channel_1*channel_1,10))
fc_b = zero_weight((10,))
```

## 3.PyTorch Module API

本节则简单，就是实现调用pytorch封装的api实现就行了！这里我们直接来看需要填写代码处！

### 3.1 Three-Layer ConvNet

初始化conv1，w1，b1，conv2，w2，b2，以及fc及fc的w与b。

```python
self.conv1 = nn.Conv2d(in_channel,channel_1,kernel_size=5,padding=2,bias=True)  
nn.init.kaiming_normal_(self.conv1.weight)
nn.init.constant_(self.conv1.bias,0)

self.conv2 = nn.Conv2d(channel_1,channel_2,kernel_size=3,padding=1,bias=True)  
nn.init.kaiming_normal_(self.conv2.weight)
nn.init.constant_(self.conv1.bias,0)

self.fc = nn.Linear(channel_2*32*32,num_classes)
nn.init.kaiming_normal_(self.fc.weight)
nn.init.constant_(self.fc.bias,0)
```

前向传播

调用API实现即可！

```python
relu1 = F.relu(self.conv1(x))
relu2 = F.relu(self.conv2(relu1))
scores = self.fc(flatten(relu2))
```

### 3.2 Train a Three-Layer ConvNet

训练三层卷积神经网络！

```python
model = ThreeLayerConvNet(3,channel_1,channel_2,10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

## 4.PyTorch Sequential API

Pytorch中提供了容器模块，可以将上述model封装起来。

### 4.1 Three-Layer ConvNet

就直接用nn.Sequential包装起来即可！其余的基本一样！

```python
model = nn.Sequential(
    nn.Conv2d(3,channel_1,kernel_size=5,padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_1,channel_2,kernel_size=3,padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2*32*32, 10),
)                   
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)
```

## 5. CIFAR-10 open-ended challenge

这一节则是随意的调用api去实现自己的网络架构，然后去训练模型，使得自己的模型要在测试集上分数高于70%!

模型架构：

- [conv-bn-relu-pool]x3 -> [affine]x1 -> [softmax or SVM]

```python
layer1 = nn.Sequential(
    nn.Conv2d(3, 30, kernel_size=5, padding=2),
    nn.BatchNorm2d(30),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
layer2 = nn.Sequential(
    nn.Conv2d(30, 50, kernel_size=3, padding=1),
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
layer3 = nn.Sequential(
    nn.Conv2d(50, 100, kernel_size=3, padding=1),
    nn.BatchNorm2d(100),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
fc = nn.Linear(100*4*4, 10)
model = nn.Sequential(
    layer1,
    layer2,
    layer3,
    Flatten(),
    fc
)
learning_rate = 1e-3
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
```

最后跑分：

```
best_model = model
check_accuracy_part34(loader_test, best_model)
```

结果为：

```python
Checking accuracy on test set
Got 7712 / 10000 correct (77.12)
```

