# Image Captioning with RNNs

## 0.导语

终于来到最后一个作业assignment3，这次主要学习RNN或LSTM时序模型！有关什么是RNN以及LSTM的学习，在后面会出相应的文章解释，本节则是针对cs231n上Image Caption做的一个实践及学习代码的详解流程。下面一起来完成这个作业吧！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.下载数据集

在做这一节作业的时候，先下载assignment3，并运行D:\Jupter\assignment3\cs231n\datasets下面的get_assignment3_data.sh脚本，然后再去进行本节作业，完成RNN_Captioning.ipynb。

```python
./get_assignment3_data.sh
```

如果你的电脑是win系统，如何实现上述操作，你如果用的git，提示wget命令没找到你，如何实现了？看下面解决方案！

**win 10上如何嵌入linux的wget命令？**

>https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058
>
>https://superuser.com/questions/1075437/mingw64-bash-wget-command-not-found
>
>https://eternallybored.org/misc/wget/

这里放出实现这个的两个链接，简单说一下，下载git，然后安装上面链接中的wget，如果git里面有就不需要了，第一个链接需要墙(你懂得！)。下载wget流程如下 ：

- Download the lastest wget binary for windows from [eternallybored](https://eternallybored.org/misc/wget/) (they are available as a zip with documentation, or just an exe)
- If you downloaded the zip, extract all (if windows built in zip utility gives an error, use [7-zip](http://www.7-zip.org/)).
- Rename the file `wget64.exe` to `wget.exe` if necessary.
- Move `wget.exe` to your `Git\mingw64\bin\`.

## 2.Look at the data

为了加载HDF5文件，我们需要安装h5py，如果你的电脑已经装上了这个包，并且后面数据运行没错，说明正常，如果装上了这个包，可是后面也运行出粗，则可能是anaconda的conda安装与pip安装冲突问题，那么需要pip先卸载，在用conda重装，重启jupter即可完美解决！

在Look at th data这一节，当运行这节代码时会报如下错误，什么问题导致的呢？

![](https://github.com/Light-City/images/blob/master/tmlp.png?raw=true)

结果手动去删除的时候，发现文件在运行中，自然也就删除不掉了，这个只是个备份文件而已，所以我们找到这个命令，发现在：/assignment3/cs231n/image_utils.py文件中，找到os.remove(fname)，并注释掉，然后再次运行就可以了！

## 3.Vanilla RNN

### 3.1 step forward

看下面图片所示，图片来源于cs231n2018ppt:

![](https://github.com/Light-City/images/blob/master/vanilla.png?raw=true)

我们需要根据公式完成，RNN前向传播(cs231n/rnn_layers.py)，实现如下：

输入:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

返回：

 - next_h: Next hidden state, of shape (N, H)
 - cache: Tuple of values needed for the backward pass.

```python
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    #     pass
    ##############################################################################
    next_h=np.tanh(prev_h.dot(Wh)+x.dot(Wx)+b)
    cache=(x,next_h,prev_h,Wx,Wh)
    ##############################################################################
    return next_h, cache
```



### 3.2 step backward

输入：
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

输出：

    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)

反向传播求梯度一个难点在于注意 tanh(x),tanh(x) 的导数是 

`(1−tanh(x)*tanh(x))`

```python
def rnn_step_backward(dnext_h, cache):
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    #     pass
    ##############################################################################
    x,next_h,prev_h,Wx,Wh=cache
    # next_h shape(N.H)
    # x shape(N,D)
    # Wx shape(D,H)
    # tmp shape(N,H)
    # d(tanh) = 1 - tanh * tanh
    tmp=(1-next_h*next_h)*dnext_h
    dx=tmp.dot(Wx.T)
    # shape(N,H)
    # Wh shape(H,H)
    dprev_h=tmp.dot(Wh.T)
    # dWx(D,H)
    # x (N,D)
    dWx=x.T.dot(tmp)
    # dWh(H,H)
    dWh=prev_h.T.dot(tmp)
    db=np.sum(tmp,axis=0)
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db
```

### 3.3 forward


输入：
- x: Input data for the entire timeseries, of shape (N, T, D).
- h0: Initial hidden state, of shape (N, H)
- Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
- Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
- b: Biases of shape (H,)

输出：
- h: Hidden states for the entire timeseries, of shape (N, T, H).
- cache: Values needed in the backward pass


```python
def rnn_forward(x, h0, Wx, Wh, b):
    ##############################################################################
    N, T, D = x.shape
    N, H = h0.shape

    h = np.zeros((N, T, H))
    prev_h = h0

    cache = {}

    for t in range(T):
        prev_h, cache_t = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = prev_h
        cache[t] = cache_t
    ##############################################################################
```



### 3.4 backward

输入：
- dh: Upstream gradients of all hidden states, of shape (N, T, H). 

输出：
- dx: Gradient of inputs, of shape (N, T, D)
- dh0: Gradient of initial hidden state, of shape (N, H)
- dWx: Gradient of input-to-hidden weights, of shape (D, H)
- dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
- db: Gradient of biases, of shape (H,)

```python
def rnn_backward(dh, cache):
    ##############################################################################
    N, T, H = dh.shape
    x = cache[0][0]
    N, D = x.shape
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    dprev_h = np.zeros((N, H))
    for t in reversed(range(T)):
        # Watch out the `NOTE` for dh!
        dnext_h = dh[:, t, :] + dprev_h
        dx[:, t, :], dprev_h, dWx_tmp, dWh_tmp, db_tmp = rnn_step_backward(dnext_h, cache[t])
        dWx += dWx_tmp
        dWh += dWh_tmp
        db += db_tmp

    dh0 = dprev_h
    ##############################################################################
```



总结：上面先进行了单步的前向与后向，随后将其拓展到多步的前向与后向。

## 4.Word embedding

### 4.1 forward

输入：
- x: 维度为(N,T)的整数列，每一项是相应词汇对应的索引。
- W: 维度为(V,D)权值矩阵，V是词表的大小，每一列对应着一个词的向量表示

输出：
- out: 维度为(N, T, D)，由所有输入词的词向量所组成
- cache: 反向传播时需要的变量

```python
def word_embedding_forward(x, W):
    ##############################################################################
    out = W[x, :]
    cache = (W, x)
    # 上述等价于下面注释掉的代码
#     N, T = x.shape
#     V, D = W.shape
#     out = np.zeros((N, T, D))
#     for i in range(N):
#         for j in range(T):
# #             out[i, j,:] = W[x[i,j],:]
#             out[i, j] = W[x[i,j]]
    cache = (x, W)
    ##############################################################################
    return out, cache
```



### 4.2  backward

输入：
- dout: 梯度， 维度(N, T, D)
- cache: 前向传播存的变量

输出：

- dW: 词嵌入矩阵的梯度，维度 (V, D).

```python
def word_embedding_backward(dout, cache):
    dW = None
    ##############################################################################
    x,W = cache
    dW = np.zeros(W.shape)
    # 将dW在x位置上与dout相加
    np.add.at(dW,x,dout)
    # 上述代码等价于下面几行
#     N,T=x.shape
#     dW = np.zeros(W.shape)
#     for row in range(N):
#         for col in range(T):
#             dW[x[row,col],:] += dout[row,col,:]
    ##############################################################################
    return dW
```

## 5.RNN for image captioning

**loss完成图片注释生成系统**

完善/assignment3/cs231n/classifiers/rnn.py代码：计算训练时RNN/LSTM的损失函数。我们输入图像特征和正确的图片注释，使用RNN/LSTM计算损失函数和所有参数的梯度！
**输入：**

        - features: 输入图像特征，维度 (N, D)
             ptions: 正确的图像注释; 维度为(N, T)的整数列

**输出：**

- loss: 标量损失函数值
- grads: 所有参数的梯度

**提示：**

(1) 使用仿射变换从图像特征计算初始隐藏状态。最终输出 shape (N, H)。

(2) 用词嵌入层将captions_in中词的索引转换成词响亮，得到一个维度为(N, T, W)的数组。

(3) 使用vanilla RNN或LSTM（取决于self.cell_type）来处理输入字向量序列并为所有时间步长产生隐藏状态向量，从而产生形状（N，T，H）的数组。

(4) 使用（时间）仿射变换在每个时间步使用隐藏状态计算词汇表上的分数，给出形状（N，T，V）的数组。

(5) 使用（temporal）softmax使用captions_out计算损失，使用上面的掩码忽略输出字<NULL>的点。


```python
def loss(self, features, captions):
    # 这里将captions分成了两个部分，captions_in是除了最后一个词外的所有词，是输入到RNN/LSTM的输入；captions_out是除了第一个词外的所有词，是RNN/LSTM期望得到的输出。
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    # You'll need this
    mask = (captions_out != self._null)
    # Weight and bias for the affine transform from image features to initial
    # hidden state
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    # 词嵌入矩阵
    W_embed = self.params['W_embed']
    # RNN/LSTM参数
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    # 每一隐藏层到输出的权值矩阵和偏差
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    loss, grads = 0.0, {}
    #####################################################################
    # 实现第一步
    hidden_init, cache_init = affine_forward(features, W_proj, b_proj)
    # 实现第二步
    captions_in_init, cache_embed = word_embedding_forward(captions_in,W_embed)
    # 实现第三步
    if self.cell_type == 'rnn':
        hidden_rnn, cache_rnn = rnn_forward(captions_in_init, hidden_init, Wx, Wh, b)
    else:
        hidden_rnn, cache_rnn = lstm_forward(captions_in_init, hidden_init, Wx, Wh, b)
    # 实现第四步
    scores, cache_scores = temporal_affine_forward(hidden_rnn, W_vocab, b_vocab)
    # 实现第五步
    loss, dscores = temporal_softmax_loss(scores, captions_out, mask)
    # 实现反向传播
    dhidden_rnn, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dscores, cache_scores)
    if self.cell_type == 'rnn':
        dcaptions_in_init, dhidden_init, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dhidden_rnn, cache_rnn)
    else:
        dcaptions_in_init, dhidden_init, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dhidden_rnn, cache_rnn)
    grads['W_embed'] = word_embedding_backward(dcaptions_in_init, cache_embed)
    dfeatures, grads['W_proj'], grads['b_proj'] = affine_backward(dhidden_init, cache_init)
```

**图片注释采样完成sample**

**输入：**

- captions: 输入图像特征，维度 (N, D)
- max_length: 生成的注释的最长长度

**输出：**

- captions: 采样得到的注释，维度(N, max_length), 每个元素是词汇的索引

**提示：**

(1) 使用学习的单词嵌入嵌入前一个单词。
(2) 使用先前的隐藏状态和嵌入的当前字进行RNN步骤以获得下一个隐藏状态。
(3) 将学习的仿射变换应用于下一个隐藏状态，以获得词汇表中所有单词的分数。
(4) 选择分数最高的单词作为下一个单词，将其（单词索引）写入标题变量中的相应插槽。

```python
def sample(self, features, max_length=30):
    hidden_init, _ = affine_forward(features, W_proj, b_proj)
    # 实现(1)
    start_word_embed, _ = word_embedding_forward(self._start, W_embed)
    hidden_curr = hidden_init
    cell_curr = np.zeros_like(hidden_curr)
    word_embed = start_word_embed
    for step in range(max_length):
        if self.cell_type == 'rnn':
            # 实现(2)
            hidden_curr, _ = rnn_step_forward(word_embed, hidden_curr, Wx, Wh, b)
        else:
            hidden_curr, cell_curr, _ = lstm_step_forward(word_embed, hidden_curr, cell_curr, Wx, Wh, b)
        # 实现(3)
        step_scores, _ = affine_forward(hidden_curr, W_vocab, b_vocab)
        # 实现(4)
        captions[:, step] = np.argmax(step_scores, axis=1)
        # word_embed作为下一次的输入
        word_embed, _ = word_embedding_forward(captions[:, step], W_embed)
```

## 6.问题

在我们当前的图像字幕设置中，我们的RNN语言模型在每个时间步长处生成一个单词作为其输出。 然而，提出问题的另一种方法是训练网络对字符（例如'a'，'b'等）进行操作而不是单词，以便在每个时间步长处，它接收前一个字符作为输入 并尝试预测序列中的下一个字符。 例如，网络可能会生成一个标题

'A'，''，'c'，'a'，'t'，''，'o'，'n'，''，'a'，''，'b'，'e'，'d“

您能描述使用字符级RNN的图像字幕模型的一个优点吗？ 你能描述一个缺点吗？ 提示：有几个有效的答案，但比较单词级和字符级模型的参数空间可能很有用。

以单词为单位的 RNN，词汇表可以很大，而且每次的输出至少能够保证是有意义的单词；而以字母为单位的 RNN，词汇表是固定大小，但是输出的范围几乎是无穷的，并且不能保证输出的组合是有意义的单词。所以一字母为单位的 RNN 效果应该不如一单词为单位的 RNN。

> 参考文章：https://blog.csdn.net/FortiLZ/article/details/80935136

## 7.作者的话

本篇文章阐述了cs231n中assignment3的第一个作业，希望能够对各位有所收获！