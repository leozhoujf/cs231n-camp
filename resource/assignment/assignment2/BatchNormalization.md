# 手推Assignment2的BN反向传播

## 0.说在前面

或许有些人对于上一节说的BN推导没有深入理解，那么本节则从一篇非常好的论文中来实践带大家手推一遍，与此同时，完成Week6的作业！

对于本节内容，请各位拿出纸与笔，下面一起来实战吧！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.Paper

这里推荐一篇论文，非常值得看！也非常经典，看不懂的，可以到网上去看，有很多大佬做了相应的阐述。这里放出论文地址：

> https://arxiv.org/abs/1502.03167

或者直接搜：**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**

## 2.手推

这里直接放出论文中的相关公式，以及自己的手推BN反向传播！

![](https://github.com/Light-City/images/blob/master/bn.png?raw=true)

## 3.任务及BN实现

【**任务**】

本节作业是

- 完成 assignment2 中 `BatchNormalization.ipynb`
- 完成 assignment2 中 `Dropout.ipynb`

第二个dropout已经在上节给出代码了，直接运行就可以了，只需要做的就是里面的cell问题即可！

今天这篇重点讲解本节难点：也就是第一个任务，完成BatchNormalization！

由于上一节已经介绍了反向传播，代码也给了，但是在这里缺少实现batchnorm_backward_alt，于是我们今天重点就放到了完成这个代码上面，同时温故上一节BN方向传播公式，并给出今日这个未完成方法的完整手推及实现！

【**BN实现**】

有关BN的手推上面给出了图片解释，这里只需要完成相应代码即可！

实现在layers.py找到下面方法完善即可！

```python
def batchnorm_backward_alt(dout, cache):
    dx, dgamma, dbeta = None, None, None
    x_normalized, gamma, beta, sample_mean, sample_var, x, eps = cache
    N, D = x.shape
    dx_normalized = dout * gamma       # [N,D]
    x_mu = x - sample_mean             # [N,D]
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)    # [1,D]
    dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    dx = (1./N)*sample_std_inv*(N*dx_normalized-np.sum(dx_normalized,axis=0,keepdims=True)*x_normalized-np.sum(dx_normalized,axis=0,keepdims=True))
    return dx, dgamma, dbeta
```

这里给出为什么这个任务需要完成这个呢？

对于cs231n的课程作业，有个很好的优点，就是注释详细，不懂的直接看，下面给出英文及中文解释！

英文解释：

After doing so, implement the simplified batch normalization backward pass in the function `batchnorm_backward_alt` and compare the two implementations by running the following. Your two implementations should compute nearly identical results, but the alternative implementation should be a bit faster.

 在这样做之后，在函数`batchnorm_backward_alt`中实现简化的批量规范化反向传递，并通过运行以下命令来比较这两个实现。您的两个实现应该计算几乎相同的结果，但替代实现应该更快一点！

哈哈，这个是我谷歌翻译的，我就直接照搬过来了，想表达的意思就是最后的求导尽量用原来公式来代替！这也就是方法里面有个alt，这个英文全拼就是alternative！

## 4.作者的话

如果您觉得本公众号对您有帮助，欢迎转发，支持！！谢谢！！有关更多内容，请关注本公众号：作业详解系列！！！

