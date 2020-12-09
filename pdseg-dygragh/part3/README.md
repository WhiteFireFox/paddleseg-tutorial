# **Windows：如何利用PaddleSeg做一个完整的项目(动态图)③**

# **PaddleSeg简介**

&emsp;&emsp;<font size=4>PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。</font><br><br>
&emsp;&emsp;<font size=4>文章将在 **“[Windows：如何利用PaddleSeg做一个完整的项目(动态图)②](https://github.com/WhiteFireFox/paddleseg-tutorial/new/main/pdseg-dygragh/part1)”** 基础上，继续以一个例子为说明，来谈如何利用PaddleSeg完成一个项目。</font>

# **项目目录**

&emsp;&emsp;<font size=4>6、模型简介</font><br><br>
&emsp;&emsp;<font size=4>7、模型评估</font><br><br>
&emsp;&emsp;<font size=4>8、模型导出</font><br><br>
&emsp;&emsp;<font size=4>9、python进行单张/多张图片的预测(方案一)</font><br><br>
&emsp;&emsp;<font size=4>10、python进行单张/多张图片的预测(方案二)</font><br><br>

# **6、模型简介**

&emsp;&emsp;<font size=4>训练完成后，根据训练时在命令行flags处，指定了的模型保存的文件目录，找到网络模型保存的地方。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/f9c0ff1e244a41be9af70d4d69eabf3cb5d2f55ea95745d2924ed55a9819c651)<br><br>
&emsp;&emsp;<font size=4>模型展示(model.pdparams保存模型参数，model.pdopt保存优化器参数)：</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/f97fe0273d2c4f9d82b47c4483060ed07380cf907ee740fca7e4d703e4c51bdb)<br><br>
&emsp;&emsp;<font size=4>我们将基于这些保存的模型进行下面的几步。</font>

# **7、模型评估**

&emsp;&emsp;<font size=4>在cmd命令行中输入代码，对模型进行评估。</font>


```python
python ./PaddleSeg/dygraph/val.py --config ./PaddleSeg/dygraph/configs/unet/unet.yml --model_dir ./output/iter_5
```



| 参数      | 作用               |
| --------- | ------------------ |
| model_dir | 模型保存的文件目录 |



&emsp;&emsp;<font size=4>示例：</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/f6f3c18d93b54b87a70e85a68f9deac27d358428c329449cb89ba8f638d24e56)

# **8、模型导出**

&emsp;&emsp;<font size=4>动态图保存模型时候没有model.pdmodel保存网络模型结构，只保存了model.pdparams模型参数，model.pdopt优化器参数，可能是本人才疏学浅，暂时没有找到方法导出。</font>

# **9、python进行单张/多张图片的预测(方案一)**

&emsp;&emsp;<font size=4>直接使用网络刚训练出来的模型进行预测。</font><br><br>
&emsp;&emsp;<font size=4>`./PaddleSeg/dygraph/`目录下的`predict.py`即为预测所需代码。</font>

## **预测命令的flags**


| flags      | 作用                                     |
| ---------- | ---------------------------------------- |
| model_dir  | 模型所在位置                             |
| image_path | 需要预测的图片所在的位置                 |
| save_dir   | 指定预测结果(即语义分割图)保存的文件地址 |


## **开始预测**

&emsp;&emsp;<font size=4>代码示例：</font>


```python
python ./PaddleSeg/dygraph/predict.py --config ./PaddleSeg/dygraph/configs/unet/unet.yml --model_dir ./output/iter_5 --image_path ./test/N0005.jpg --save_dir ./result
```

<br><br>&emsp;&emsp;<font size=4>代码运行示例：</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/60ecab7fcd0e4bc9a8bb304f4fa0bcf9a45717c0a86b46149beb389c9dfed716)<br><br>
&emsp;&emsp;<font size=4>执行完后在reslut文件夹下的pseudo_color_prediction文件夹里，里面存放着测试集图片的预测结果。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/f0ef9ac90bc8468eaf2194ac39ae2f40b59305d1908142aaa71606b96f15db17)<br><br>
&emsp;&emsp;<font size=4>执行完后在reslut文件夹下的added_prediction文件夹里，里面存放着测试集图片与测试集图片的预测结果叠加而成的图片。</font>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/670531a223394a49b517928c856b485ef20d8fcee95f49fe8f083fa80fbda0e3)

# **10、python进行单张/多张图片的预测(方案二)**

&emsp;&emsp;<font size=4>使用导出的模型进行预测。</font><br><br>
&emsp;&emsp;<font size=4>方案待定。</font><br><br>
