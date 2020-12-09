# **Windows：如何利用PaddleSeg做一个完整的项目③**

# **PaddleSeg简介**

&emsp;&emsp;<font size=4>PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。</font><br><br>
&emsp;&emsp;<font size=4>文章将在 **“[如何利用PaddleSeg做一个完整的项目②](https://github.com/WhiteFireFox/paddleseg-tutorial/new/main/pdseg)”** 基础上，继续以一个例子为说明，来谈如何利用PaddleSeg完成一个项目。</font>

# **项目目录**

&emsp;&emsp;<font size=4>6、模型简介</font><br><br>
&emsp;&emsp;<font size=4>7、模型评估</font><br><br>
&emsp;&emsp;<font size=4>8、模型导出</font><br><br>
&emsp;&emsp;<font size=4>9、python进行单张/多张图片的预测(方案一)</font><br><br>
&emsp;&emsp;<font size=4>10、python进行单张/多张图片的预测(方案二)</font><br><br>

# **6、模型简介**

&emsp;&emsp;<font size=4>训练完成后，根据配置文件.yaml中模型保存的文件目录，找到网络模型保存的地方。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/588dce66eb0e4aaabe68f73025affc4504a8b8ddd3ed467cb05fa447d42a23d4)<br><br>
&emsp;&emsp;<font size=4>模型展示(model.pdmodel保存网络模型，model.pdparams保存模型参数，model.pdopt保存优化器参数)：</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/9b69e6063c8344019e37388a685abea5ce0f5a024b034318822d94b748ea8075)<br><br>
&emsp;&emsp;<font size=4>我们将基于这些保存的模型进行下面的几步。</font>

# **7、模型评估**

&emsp;&emsp;<font size=4>在cmd命令行中输入代码，对模型进行评估。</font><br><br>


```python
# 使用CPU
python ./PaddleSeg-release-v0.6.0/pdseg/eval.py --cfg ./PaddleSeg-release-v0.6.0/configs/unet_optic.yaml TEST.TEST_MODEL ./saved_model/unet_optic/final

# 使用GPU
python ./PaddleSeg-release-v0.6.0/pdseg/eval.py --use_gpu --cfg ./PaddleSeg-release-v0.6.0/configs/unet_optic.yaml TEST.TEST_MODEL ./saved_model/unet_optic/final
```

&emsp;&emsp;<font size=4>示例：</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/2541822d6f8d47fcaa87145673290e563ec2cce452f04fa39ff83a1c031eb8eb)

# **8、模型导出**

&emsp;&emsp;<font size=4>在cmd命令行中输入代码，对模型进行导出。</font><br><br>


```python
python ./PaddleSeg-release-v0.6.0/pdseg/export_model.py --cfg ./PaddleSeg-release-v0.6.0/configs/unet_optic.yaml TEST.TEST_MODEL ./saved_model/unet_optic/final
```

&emsp;&emsp;<font size=4>预测模型会导出到`freeze_model`目录下。通常包含model、params和deploy.yaml三个文件。</font><br><br>
&emsp;&emsp;<font size=4>├── model #  模型文件</font><br>
&emsp;&emsp;<font size=4>├── params # 参数文件</font><br>
&emsp;&emsp;<font size=4>└── deploy.yaml # 配置文件，用于C++或Python预测</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/01ad4c1b0f7f429ebcbbdba52737f99796b9cd2a130d44b28a33cc536a58c91a)

# **9、python进行单张/多张图片的预测(方案一)**

&emsp;&emsp;<font size=4>直接使用网络刚训练出来的模型进行预测。</font><br><br>
&emsp;&emsp;<font size=4>`./PaddleSeg-release-v0.6.0/pdseg/`目录下的`vis.py`即为预测所需代码。</font><br><br>
&emsp;&emsp;<font size=4>预测前需要做好两件事：</font>

## **①生成test_list.txt告诉网络需要预测的图片的位置**

![](https://ai-studio-static-online.cdn.bcebos.com/f7e0bc0fb5964978b948d3bca99782787a8f5f7246a84c2e9c425a5aa036f13c)

## **②在配置文件中加上test_list.txt**

&emsp;&emsp;<font size=4>如同所示。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/f3090c0159494ec7be0c0fddc1a8007e958fe6b506914ff18d60754584d254d3)

## **开始预测**

&emsp;&emsp;<font size=4>代码示例：</font><br><br>


```python
# 使用CPU
python ./PaddleSeg-release-v0.6.0/pdseg/vis.py --cfg ./PaddleSeg-release-v0.6.0/configs/unet_optic.yaml TEST.TEST_MODEL ./saved_model/unet_optic/final

# 使用GPU
python ./PaddleSeg-release-v0.6.0/pdseg/vis.py --use_gpu --cfg ./PaddleSeg-release-v0.6.0/configs/unet_optic.yaml TEST.TEST_MODEL ./saved_model/unet_optic/final
```

&emsp;&emsp;<font size=4>执行完后会在主目录下产生一个visual文件夹，里面存放着测试集图片的预测结果。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/3e53190c58e24c058d59214a7c0a8a227bb1f2690b2d4c84b0a3595cbeeed9bc)

# **10、python进行单张/多张图片的预测(方案二)**

&emsp;&emsp;<font size=4>使用导出的模型进行预测。</font><br><br>
&emsp;&emsp;<font size=4>在`./PaddleSeg-release-v0.6.0/deploy/python`目录下的`infer.py`即为预测所需代码。</font><br><br>
&emsp;&emsp;<font size=4>参数配置说明：</font><br>

| 参数      | 是否必须 | 模型配置的Yaml文件路径 |
| --------- | -------- | ---------------------- |
| conf      | Yes      | 模型配置的Yaml文件路径 |
| input_dir | Yes      | 需要预测的图片目录     |

<br>&emsp;&emsp;<font size=4>代码示例：</font><br><br>


```python
python ./PaddleSeg-release-v0.6.0/deploy/python/infer.py --conf=./freeze_model/deploy.yaml --input_dir=./test/
```

&emsp;&emsp;<font size=4>执行完后，对于图片`a.jpg`, 预测mask存在`a_jpg.png`中，而可视化结果则在`a_jpg_result.png`中</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/c6966a5e31f343458dfce34be3ad158e85f6277602004ca1acff23fe8f8880b0)
