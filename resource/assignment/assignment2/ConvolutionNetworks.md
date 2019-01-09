# 卷积神经网络

## 0.说在前面

今天来个比较嗨皮的，那就是大家经常听到的卷积神经网络，也就是Convolutional Neural Networks，简称CNNs！

这次完成的任务是Week7！前方高能预警，这节内容较多，希望心平气和再看，信息量庞大！收获会很多，下面一起来搞事!

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.卷积神经网络

为了更好的理解后面的代码实现部分，这里再次回顾一下卷积神经网络的构成，主要由三种类型的层来构成：**卷积层**，**汇聚层**和**全连接层**！

### 1.1 卷积层

为了更好的理解卷积神经网络，这里给出一张图：

![](https://github.com/Light-City/images/blob/master/inpiutxcs.png?raw=true)

这里给出的是静态图，想看动态图的可以去官网看，这里给出官网地址：

> http://cs231n.github.io/convolutional-networks/

假设卷积层中神经元的感受野尺寸为F，步长为S，零填充的数量为P，则通过函数计算，最终输出数据的空间尺寸为

`(W-F+2*P)/S+1`，

比如上图：

我们看到这个图采用了零填充，通过公式P=(F-1)/2计算出得，P=(3-1)/2=1。

那么输出尺寸计算为`(5-3+2*1)/2+1=3`，所以输出尺寸为3！

**零填充得目的是保证输入和输出数据题有相同得空间尺寸！**

那么怎么计算最后得输出值？

每个元素都是先通过蓝色的输入数据和红色的滤波器对应元素相乘，然后求其总和，最后加上偏差得来。

以图中输出得5为例： 

```python
第一个通道相乘结果
0  0  0
0  0  0       求和2
0  0  2
第二个通道相乘结果                       
0  0  0                         三个总和为4，再加偏差b，得到5
0  0  1       求和1
0  0  0 
第三个通道相乘结果
0  0  0
0  0  2       求和1
0 -1  0
```



### 1.2 汇聚层

汇聚层最常用得是Max-Pooling，如下图所示：

![](https://github.com/Light-City/images/blob/master/max_poll.png?raw=true)

汇聚层最常见得形式是使用尺寸2x2的滤波器，以步长为2来对每个深度切片进行降采样，将其中75%的激活信息都丢掉。因为每次取感受野扫过得最大值作为输出，最后只有1/4得激活信息，所以丢掉了75%激活信息！

### 1.3 全连接层

在本节实战中，我们会用到之前编写得仿射层前向与后向传播！仿射层是神经网络中的一个全连接层。仿射的意思是前一层中的任一神经元都连接到后一层中的任一个神经元。注意，全连接层的最后一层就是输出层；除了最后一层，其它的全连接层都包含激活函数。

前面几个概念阐述完毕，下面一起来实战吧！



## 2.卷积层实现

### 2.1 前向传播

卷积层与反向传播都在layers.py中去完成！

【**目标**】

实现卷积层得前向传播

【**输入**】

 输入由N个数据点组成，每个数据点具有C通道，高度H和宽度W.我们使用F不同的滤波器对每个输入进行卷积，其中每个滤波器跨越所有C通道并具有高度HH和宽度WW。

下面是注释中输入数据要求：

```python
 Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
```

【**输出**】

下面是注释所返回得数据要求：

```python
Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
```

【**实现**】

实现之前先来讲一下：**零填补问题**！

```python
x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant',constant_values=0)
```

输入数据的x维度为：(N, C, H, W)。

我们实际需要填充的是(H,W)维度，所以实现的时候是前面两个均为0，后面填充均为pad！

模拟实验，这里我假设为二维数组：

```python
import numpy as np
x = np.array(np.arange(1,7)).reshape(2,3)
x
pad = 1
x_pad = np.pad(x,((pad,pad),(pad,pad)),mode='constant',constant_values=0)
x_pad
```

输出：

```python
array([[1, 2, 3],
       [4, 5, 6]])
array([[0, 0, 0, 0, 0],
       [0, 1, 2, 3, 0],
       [0, 4, 5, 6, 0],
       [0, 0, 0, 0, 0]])
```

经过0填补后，就变为我们所需要的填补矩阵！

由于前面两个维度(N,C)不需要填补，所以直接为(0,0)即可！

现在就有人问了，(pad,pad)这个元组表达啥意思？

举个例子：第一行 1 2 3 上面填充的0就是第一个元组的第一个pad，那么最后一行4 5 6下面填充的0就是第一个元组的第二个pad，同理后面的元组为左右方向！

填充的数据为0是通过constant_values来控制的，你也可以填充其他数据，比如1 ，此时就可以通过这个参数设置，而mode的话，这里不多说，具体大家看api，到后面会详细参数这个参数的意思！

代码实现思路为：

首先获取输入数据x与w的所有维度，通过上面的卷积层讲解，大家肯定会计算输出数据以及如何计算输出数据的尺寸！

由于上面研究X与W维度都相同，而在这里则不同！这就是现实意义与理想的区别！

最后的四层for循环目的是遍历N个数据，对每个数据进行卷积输出计算，具体的输出计算大家可以将代码与上面的例子结合起来！

```python
def conv_forward_naive(x, w, b, conv_param)
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride,pad= conv_param['stride'],conv_param['pad']
    N, C, H, W=x.shape
    F, C, HH, WW=w.shape
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant',constant_values=0)
    new_H = 1 + (H + 2 * pad - HH) // stride
    new_W = 1 + (W + 2 * pad - WW) // stride
    s = stride
    out = np.zeros((N,F,new_H,new_W))
    for n in range(N):
        for f in range(F):
            for j in range(new_H):
                for k in range(new_W):
                    out[n, f, j, k] = np.sum(x_pad[n, :, j*s:j*s+HH, k*s:k*s+WW]*w[f, :]) + b[f]
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache
```

### 2.2 反向传播

【**目标**】

实现卷积层的反向传播

【**输入**】

```python
Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
```

【**输出**】

```python
Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
```

【**实现**】

反向传播与前向传播计算非常相似！大家直接看代码即可，如果不懂，留言！

```python
def conv_backward_naive(dout, cache):
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
#     pass
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W=x.shape
    F, C, HH, WW=w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    new_H = 1 + (H + 2 * pad - HH) // stride
    new_W = 1 + (W + 2 * pad - WW) // stride
    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant',constant_values=0)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    s = stride
    for i in range(N):       # ith image    
        for f in range(F):   # fth filter        
            for j in range(new_H):            
                for k in range(new_W):                
                    window = x_pad[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]                
                    dw[f] += window * dout[i, f, j, k]                
                    dx_pad[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    ###########################################################################
    return dx, dw, db
```

## 3.汇聚层

### 3.1 前向传播

【**目标**】

实现max-pooling的前向传播

【**输入**】

```python
 Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
```

【**输出**】

```python
Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
```

【**实现**】

基本操作同上，不同之处在于每次获取一个感受野大小的窗体，从这个窗体中每次取出最大值，就是max-pooling所要实现的！

```python
def max_pool_forward_naive(x, pool_param):
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
#     pass
    ###########################################################################
    N, C, H, W = x.shape
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    new_H = 1 + (H - HH) // s
    new_W = 1 + (W - WW) // s
    out = np.zeros((N, C, new_H, new_W))
    for i in range(N):    
        for j in range(C):        
            for k in range(new_H):            
                for l in range(new_W):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s] 
                    out[i, j, k, l] = np.max(window)
    ###########################################################################
    cache = (x, pool_param)
    return out, cache
```

### 3.2 反向传播

【**目标**】

【**输入**】

```python
Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
```

【**输出**】

```python
Returns:
    - dx: Gradient with respect to x
```

【**实现**】

这里的反向传播方法在前面其实说过，就在任务一中，也明确举例子了的，所以这里我只简单提一下，核心代码为：

`window == np.max(window)`，而这句是通过得到布尔矩阵，通过这个布尔矩阵来计算反向传播！

```python
def max_pool_backward_naive(dout, cache):
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
#     pass
    ###########################################################################
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    new_H = 1 + (H - HH) // s
    new_W = 1 + (W - WW) // s
    dx = np.zeros_like(x)
    for i in range(N):    
        for j in range(C):        
            for k in range(new_H):            
                for l in range(new_W):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]                    
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == np.max(window)) * dout[i, j, k, l]
    ###########################################################################
    return dx
```

## 4.组合层

由于原本代码已经实现了这一块，所以不多做也是，组合层在layer_utils.py中，主要完成了conv - relu - 2x2 max pool组合！

## 5.三层卷积神经网络

### 5.1 架构

首先来了解一下三层卷积神经网络的架构：

conv - relu - 2x2 max pool - affine - relu - affine - softmax

### 5.2 类构造方法

【**注释**】

下面给出输入的注释：

```python
Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
```

下面给出注释中的提示：

Initialize weights and biases for the three-layer convolutional network. Weights should be initialized from a Gaussian centered at 0.0 with standard deviation equal to weight_scale; biases should be,initialized to zero. All weights and biases should be stored in the dictionary self.params. Store weights and biases for the convolutional layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the hidden affine layer, and keys 'W3' and 'b3' for the weights and biases of the output affine layer.

翻译过来就是，初始化权重与偏差，权重得服从高斯分布，偏差则是初始化为0，并要求存储权重与偏差，W1作为key存储卷积层的权重，b1作为key存储卷积层的偏差，其余的w2，b2则在隐藏的仿射层中；w3，b3则作为输出仿射层的权重与偏差！

【**实现**】

通过上面的提示，我们基本上直到该怎么做了！

```python
C, H, W = input_dim
self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
self.params['b1'] = np.zeros((1, num_filters))
# 这里为社么是除以4，请看上面的max-pooling解释！最终只保留了1/4的数据！
self.params['W2'] = weight_scale * np.random.randn(num_filters*H*W//4, hidden_dim)
self.params['b2'] = np.zeros((1, hidden_dim))
self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
self.params['b3'] = np.zeros((1, num_classes))
```
### 5.3计算损失

#### 5.3.1 前向传播

具体看注释解释：

```python
############################################################################
# 计算前向传播
# 前向计算三层卷积的loss
# 首先来计算conv - relu - 2x2 max pool
'''
查看layer_utils.py，里面有个conv_relu_pool_forward函数，将conv - relu - 2x2 max pool
组合了起来！
'''
out1,cache1 = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
# 紧接着计算affine - relu
out2,cache2 = affine_relu_forward(out1,W2,b2)
# 计算affine - softmax(affine_forward函数在layers.py中)
scores, cache3 = affine_forward(out2, W3, b3)
############################################################################
```



#### 5.3.2 反向传播

具体看注释解释：

```python
############################################################################
# 计算反向传播
# 反向传播求梯度
# softmax_loss函数在layers.py中
sm_loss, dscores = softmax_loss(scores, y)
dout2, dW3, db3 = affine_backward(dscores, cache3)
dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)
# 添加正规化系数
dW1 += self.reg * W1
dW2 += self.reg * W2
dW3 += self.reg * W3
reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])
loss = sm_loss + reg_loss
grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
############################################################################
```



## 6.Spatial batch normalization

### 6.1 要求解读

我们在运行ConvolutionalNetworks.ipynb的时候，发现如下图的要求：

![](https://github.com/Light-City/images/blob/master/piliang.png?raw=true)

于是我们按照下面要求完成前向传播与反向传播!

### 6.2 前向传播

【**目标**】

计算spatial batch normalization前向传播

【**输入**】

```python
 Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
```

【**输出**】

```python
Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
```

【**实现**】

TODO中说到不能超过5行去实现！具体的解释看如下代码注释：

```python
###########################################################################
N,C,H,W=x.shape
# 通过transpose(0,2,3,1)，将shape变为(N,H,W,C)，然后在reshape为bn前向传播传递的x的shape
new_x=x.transpose(0,2,3,1).reshape(N*H*W,C)
# out的shape为(N*H*W,C)
out, cache = batchnorm_forward(new_x, gamma, beta, bn_param)
# 先reshape一下，从二维变为四维，在调整位置进行转置，变为shape=(N,C,H,W)
out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
###########################################################################
```



### 6.3 反向传播

【**目标**】

计算spatial batch normalization反向传播

【**输入**】

```python
Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
```

【**输出**】

```python
Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
```

【**实现**】

TODO：You can implement spatial batch normalization by calling the vanilla version of batch normalization you implemented above. 

这个反向传播就是直接带调用之前写的BN反向传播函数来完成！

```python
##########################################################################
N, C, H, W = dout.shape
new_dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)
dx, dgamma, dbeta = batchnorm_backward(new_dout, cache)
dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
###########################################################################
```



## 7.Group Normalization

### 7.1 什么是Group Normalization？

![](https://github.com/Light-City/images/blob/master/group.png?raw=true)

从左到右一次是BN，LN，IN，GN 众所周知，深度网络中的数据维度一般是[N, C, H, W]或者[N, H, W，C]格式，N是batch size，H/W是feature的高/宽，C是feature的channel，压缩H/W至一个维度，其三维的表示如上图！

上图形象的表示了四种norm的工作方式：

- BN在batch的维度上norm，归一化维度为[N，H，W]，对batch中对应的channel归一化；
- LN避开了batch维度，归一化的维度为[C，H，W]；
- IN 归一化的维度为[H，W]；
- **而GN介于LN和IN之间，其首先将channel分为许多组（group），对每一组做归一化，及先将feature的维度由[N, C, H, W]reshape为[N, G，C//G , H, W]，归一化的维度为[C//G , H, W]**

上面阐述引用来自：

> 全面解读Group Normalization-（吴育昕-何恺明 ）
>
> https://zhuanlan.zhihu.com/p/35005794

### 7.2 前向传播

【**输入**】

```python
Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability
```

【**输出**】

```python
Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
```

【**实现**】

下面这个代码类似于BN！修改就是维度发生了变化，具体解释看上节！

```python
###########################################################################
N, C, H, W = x.shape
# shape(C//G*H*W,N*G)
x = np.reshape(x, (N*G, C//G*H*W)).T
# mini-batch mean miu_B (1,N*G)
sample_mean = np.mean(x,axis=0,keepdims=True)
# mini-batch variance sigema_square (1,N*G)
sample_var = np.var(x,axis=0,keepdims=True)
# shape(C//G*H*W,N*G)
x_normalize = (x-sample_mean)/np.sqrt(sample_var+eps)
# Transform x_normalize and reshape
x_normalize = np.reshape(x_normalize.T, (N, C, H, W))
# scale and shift
out = gamma[np.newaxis, :, np.newaxis, np.newaxis] * x_normalize + beta[np.newaxis, :, np.newaxis, np.newaxis]
cache=(x_normalize,gamma,beta,sample_mean,sample_var,x,eps,G)

###########################################################################
```

### 7.3 反向传播

【**输入**】

```python
Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
```

【**输出**】

```python
Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
```

【**实现**】

这里的实现主要是基于batchnorm_backward_alt以及上面什么是Group Normalization来实现的！具体可以看下面注释的维度解释：

```python
###########################################################################
N, C, H, W = dout.shape
x_normalized, gamma, beta, sample_mean, sample_var, x, eps,G = cache
# shape(N,C,H,W)
dx_normalized = dout * gamma[np.newaxis, :, np.newaxis, np.newaxis] 
# Set keepdims=True to make dbeta and dgamma's shape be (1, C, 1, 1)
dgamma = np.sum(dout*x_normalized, axis=(0, 2, 3), keepdims=True)
dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

# shape(C//G*H*W,N*G)
dx_normalized = np.reshape(dx_normalized, (N*G, C//G*H*W)).T
# shape(C//G*H*W,N*G)
x_normalized = np.reshape(x_normalized, (N*G, C//G*H*W)).T
# shape(C//G*H*W,N*G)
x_mu = x - sample_mean           
# # shape(1,N*G)
sample_std_inv = 1.0 / np.sqrt(sample_var + eps)    

group_N, group_D = dx_normalized.shape
# shape(C//G*H*W,N*G)
# 见上节batchnorm_backward_alt函数详解！
dx = (1./group_N)*sample_std_inv*(group_N*dx_normalized-np.sum(dx_normalized,axis=0,keepdims=True)*x_normalized-np.sum(dx_normalized,axis=0,keepdims=True))
dx = np.reshape(dx.T, (N, C, H, W))
###########################################################################
```

