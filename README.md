DeepID1
===
Implementation of DeepID1 using tensorflow.

# 简要说明
Face Verification（人脸验证）是人脸识别中的经典任务。即给定两张图片，判定这两张图片是否属于同一个人。

[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf) 是2014年香港中文大学的汤晓鸥团队提出的一系列算法，共有三代。
这一系列基于卷积神经网路的方法，将LWF数据集上的人脸识别史无前例的推进到了99%。具体关于此方法的介绍[见此](http://blog.csdn.net/stdcoutzyx/article/details/42091205)。

本代码是 DeepID1 的 [tensorflow](https://www.tensorflow.org/) 开源实现。代码结构简单易懂。
使用了开放的 [Youtube Aligned Face](http://www.cs.tau.ac.il/~wolf/ytfaces/) 数据集，在 8:1:1 的切分下达到了 95% 的识别精度。

# 环境配置
python3: `numpy, scipy, pillow, tensorflow`

dataset: [Youtube Aligned Face](http://www.cs.tau.ac.il/~wolf/ytfaces/)

RAM: >= 12GB

# 代码概述
<pre>
.
├── crop.py           # 对数据库图片进行剪裁
├── split.py          # 将数据按照 8:1:1 划分为训练集、验证集和测试集
├── vec.py            # 将数据格式化为向量形式
├── deepid1.py        # 训练卷积神经网络，将模型参数存入 checkpoint
├── predict.py        # 根据训练出的网络模型进行人脸验证
├── checkpoint        # 存放模型参数
│   ├── 30000.ckpt
│   ├── 30000.ckpt.meta
│   └── checkpoint
├── data              # 图片数据库目录
│   ├── aligned_images_DB.tar.gz
├── log               # tensorflow 生成的 log 文件，便于可视化
│   ├── test
│   └── train
├── README.md
</pre>

# 运行步骤
`cd data; tar -zxf aligned_images_DB.tar.gz` 解压缩图片数据库

`./crop.py` 对数据库中的图片进行剪裁，由于该数据已经对齐，因此只需将人脸区域剪裁到 (55,47) 的像素即可。

`./split.py` 对剪裁后的数据库文件按照 8:1:1 的规模进行切分，分别作为训练集、验证集和测试集。每个人保留固定数目的图片（100张）进行训练。为生成测试集，对每个人构造 5 对同一个人的图片 pair，再构造 5 对不同人的图片 pair，作为测试集。

`./vec.py` 将数据格式化为向量形式，存入 `data/dataset.pkl`。便于训练时直接从该文件读取数据。

`./deepid1.py` 训练卷积神经网络，将模型参数存入 checkpoint，将运行时的统计数据存入 log。（大约需要一天左右训练完成，可以在文件内修改训练轮数和收敛速率等参数）

`tensorboard --logdir=log` 查看图模型及训练统计数据。

`./predict.py` 对测试集数据，利用训练好的网络提取特征，对比特征的 consine 相似度，得出判别是否同一个人的结果

# 实验参数
卷积神经网络完全按照 deepid1 的规格搭建，具体网络结构如下：
![](https://github.com/jinze1994/DeepID1/blob/master/data/graph.png)

一些其它的超参数设置如下：

|梯度下降方法|初始学习率|激活函数|batch size|预测向量距离|
|----------|--------|-------|----------|----------|
|AdamOptimizer| 1e-4|   relu|      1024|    cosine|

# 收敛过程
下图展示了训练过程中的收敛情况，第一幅图展示了softmax精确度随迭代次数增加的收敛情况；第二幅图展示了loss值随迭代次数的收敛情况。蓝线代表训练集的统计数据，红线代表验证集的统计数据。

![](https://github.com/jinze1994/DeepID1/blob/master/data/accu.png)

![](https://github.com/jinze1994/DeepID1/blob/master/data/loss.png)

可以看出，精确度随迭代次数呈 S 型变化，在训练集上到达了 100%，在验证集上最终到达 95% 左右。loss 也随迭代次数不断下降，根据验证集的曲线并未上升，我们可以判断出训练并未过拟合。

# 准确率
与训练过程中的softmax准确率不同，这里的准确率是指测试集上，判断一个图片 pair 是否同一个人的准确率。

|迭代轮数|准确率|
|-----|------|
|10000|94.77%|
|15000|95.92%|
|20000|95.99%|
|25000|95.83%|
|30000|95.86%|
由于小于 10000 的迭代轮数的 checkpoint 没有保存，因此懒得再跑了。可以看出，该实现在 Youtube face 数据集上的人脸验证准确率可稳定在 95% 以上。模型之前并未见过测试集中的人脸，可以看出我们的实现是成功的！
