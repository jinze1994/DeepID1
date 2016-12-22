# DeepID1
Implementation of DeepID1 using tensorflow. 

DeepID1 人脸验证算法的 tensorflow 实现

## 环境配置
python3: numpy, scipy, pillow, [tensorflow](https://www.tensorflow.org/)

dataset: [Youtube Aligned Face](http://www.cs.tau.ac.il/~wolf/ytfaces/)

RAM: >= 12GB

## 代码运行
<pre>
.
├── crop.py
├── split.py
├── vec.py
├── deepid1.py
├── predict.py
├── checkpoint
│   ├── 30000.ckpt
│   ├── 30000.ckpt.meta
│   └── checkpoint
├── data
│   ├── aligned_images_DB.tar.gz
├── log
│   ├── test
│   │   └── events.out.tfevents.1482329191.gaojun
│   └── train
│       └── events.out.tfevents.1482329190.gaojun
├── README.md
</pre>
