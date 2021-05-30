# efficientnet_pytorch
实现风格与`torchvision.models`中其他网络的类似  
(The implementation style is similar to other networks in `torchvision.models`)


## Reference
1. 论文(paper):  
[https://arxiv.org/pdf/1905.11946.pdf](https://arxiv.org/pdf/1905.11946.pdf)  

2. 代码参考(reference code):  
[https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)  

3. 使用的预训练模型来源(Source of the pre-training model used):   
[https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)  
由于我对模型进行了调整，所以我修改了原预训练模型的state_dict的key值，并进行了发布.  
Since I adjusted the model, I modified the state_dict key value of the original pretraining model and release it.  

权重见 release. 或在百度云中下载:  
链接：[https://pan.baidu.com/s/1HzAZZoJNIoii7uDsnPExHQ](https://pan.baidu.com/s/1HzAZZoJNIoii7uDsnPExHQ)  
提取码：4sb5

4. labels_map.txt来源(Source of labels_map.txt):  
[https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/simple/labels_map.txt](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/simple/labels_map.txt)

## 使用方式(How to use)
#### 1. 预测图片(Predict images)
```
python3 std_pred_image.py
```

#### 2. 简单的训练例子(Simple training examples)

```
python3 train_example.py
```

#### 3. train
```
python3 make_dataset.py
python3 train.py
python3 pred_image.py
```

## 性能

见`docs/`文件夹  

## 运行环境(environment)

torch 1.8.1  
torchvision 0.9.0  