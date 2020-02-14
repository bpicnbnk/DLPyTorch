github连不上了，其他内容之后再传吧。。。
---
基础知识了解，代码另外，这里记录新内容。
# 1.线性回归
代码问题：
1. RuntimeError: DataLoader worker (pid(s) 14224) exited unexpectedly, 两种办法
-  以下代码num_workers改为0
   ```
   data_iter = Data.DataLoader(
        dataset=dataset,            # torch TensorDataset format
        batch_size=batch_size,      # mini batch size
        shuffle=True,               # whether shuffle the data or not
        num_workers=2,              # read data in multithreading
    )
   ```
   
- 根据log
    ```
    def _check_not_importing_main():
        if getattr(process.current_process(), '_inheriting', False):
            raise RuntimeError('''
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.

            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:

                if __name__ == '__main__':
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.''')

    ```
    在程序开始添加以下代码即可。
    ```
    if __name__ == '__main__':
        freeze_support()
        # 其他代码
        ...
    ```
   
# 2.Softmax与分类模型
交叉熵: 只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。当然，遇到一个样本有多个标签时，例如图像里含有不止一个物体时，我们并不能做这一步简化。但即便对于这种情况，交叉熵同样只关心对图像中出现的物体类别的预测概率。最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

# 3.多层感知机
1. 仿射变换：又称仿射映射，是指在几何中，对一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间。 
2. 多层感知机隐藏层和输出层均是全连接层。
3. 关于激活函数的选择

    ReLu函数是一个通用的激活函数，目前在大多数情况下使用。但是，ReLU函数只能在隐藏层中使用。

    用于分类器时，sigmoid函数及其组合通常效果更好。由于梯度消失问题，有时要避免使用sigmoid和tanh函数。

    在神经网络层数较多的时候，最好使用ReLu函数，ReLu函数比较简单计算量少，而sigmoid和tanh函数计算量大很多。

    在选择激活函数的时候可以先选用ReLu函数如果效果不理想可以尝试其他激活函数。

# 4.文本预处理

文本是一类序列数据，一篇文章可以看作是字符或单词的序列，本节将介绍文本数据的常见预处理步骤，预处理通常包括四个步骤：
    读入文本
    分词
    建立字典，将每个词映射到一个唯一的索引（index）
    将文本从词的序列转换为索引的序列，方便输入模型

idx_to_text: list
text_to_idx: dcit

# 5.语言模型
## 随机采样

下面的代码每次从数据里随机采样一个小批量。其中批量大小batch_size是每个小批量的样本数，num_steps是每个样本所包含的时间步数。 在随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。
## 相邻采样

在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。


# 6.循环神经网络基础
 torch.Tensor.scatter_(dim, index, src) → Tensor
 Writes all values from the tensor src into self at the indices specified in the index tensor. For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.

For a 3-D tensor, self is updated as:
```
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```
Example:
```
>>> x = torch.rand(2, 5)
>>> x
tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
        [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
>>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
        [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
        [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])

>>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
>>> z
tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.2300]])
```

