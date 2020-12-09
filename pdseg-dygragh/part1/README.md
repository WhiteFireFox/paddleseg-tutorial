# **Windows：如何利用PaddleSeg做一个完整的项目(动态图)①**

# **PaddleSeg简介**

&emsp;&emsp;<font size=4>PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。</font><br><br>
&emsp;&emsp;<font size=4>文章将以一个例子为说明，来谈如何利用PaddleSeg完成一个项目。</font>

# **项目目录**

&emsp;&emsp;<font size=4>1、环境部署</font><br><br>
&emsp;&emsp;<font size=4>2、数据集准备①</font><br><br>

# **1、环境部署**

## **1.1、安装paddle**

&emsp;&emsp;<font size=4>打开cmd命令行，输入如下代码，安装paddle2.0.0b</font>


```python
pip install paddlepaddle==2.0.0b0 -i https://mirror.baidu.com/pypi/simple
```

![](https://ai-studio-static-online.cdn.bcebos.com/784d678adf624220bd329a0e9c9d3b63a9fb1e8983cc4b3ab500af181c8a1694)<br><br>
&emsp;&emsp;<font size=4>等待安装，当出现Successfully.....时，即安装成功。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/d99a39b46a8a45f3bba2244595a2945f15ea0514045f494ca4eec9c7b2ed15ad)

## **1.2、下载PaddleSeg**

&emsp;&emsp;<font size=4>本文提供两种下载方案。</font>

### **1.2.1、从Gitee官网直接下载(国内，较快)**

&emsp;&emsp;<font size=4>官网地址：[链接](https://gitee.com/paddlepaddle/PaddleSeg)</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/3d06f972ffa947eaa17e7c32982a188dd17bc55d65dd4dc689ab56d17dfdb7e0)<br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/8114399f50d449b4bd78c9c2102a3c947a8f1301b4ff449285b65b7d4eee61d7)

### **1.2.2、从GitHub官网直接下载(较慢)**

&emsp;&emsp;<font size=4>官网地址：[链接](https://github.com/paddlepaddle/PaddleSeg)</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/b2e096aeab5b402da6f0d289594896ee504a92eba9b843b88ea29b678653c7fc)<br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/2f2acd28e3594bcead960a1462499e0dbfc6903e94944647b5777abad1b84ef0)

## **1.3、安装其它依赖库**

&emsp;&emsp;<font size=4>将1.2中下载下来的.zip解压得到下面这个文件夹。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/4ae710abe5894e9b856b4f981b2c2d579a14b667360b4cf59d159f5287ace86e)<br><br>
&emsp;&emsp;<font size=4>依赖库文档在这个文件夹的requirements.txt中给出，可使用pip install -r requirements.txt直接安装。</font>


```python
pip install -r requirements.txt
```

&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/19f2b93f82e846ad8822f91d1b10f7d282cc86ada39346258bc7b2ad2d1e2f2f)<br><br>
&emsp;&emsp;<font size=4>等待安装，当出现Successfully.....时，即安装成功。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/32b84886405b4209899599e085a70ce983f28a0a37fc497697365b0ba5564471)

# **2、数据集准备①**

## **2.1、标注数据说明**

&emsp;&emsp;<font size=4>①PaddleSeg支持灰度图标注。</font><br><br>
&emsp;&emsp;<font size=4>②PaddleSeg也支持伪彩色图作为标注图片，在原来的单通道图片基础上，注入调色板，即在基本不增加图片大小的基础上，却可以显示出彩色的效果。</font><br><br>
&emsp;&emsp;<font size=4>灰度图与伪彩色图对比：</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/2482f442e0694399b9186b4b550e0f1d6e1a9a79ad764307bc75c6e1d9e787b1)

## **2.2、标注工具的准备**

&emsp;&emsp;<font size=4>这里我们介绍一种标注工具：LabelMe。</font><br><br>
&emsp;&emsp;<font size=4>我们在采集完用于训练、评估和预测的图片之后(即未经任何处理的原图片)，需使用数据标注工具LabelMe完成数据标注，才能丢进网络中进行训练。</font>

### **2.2.1、LabelMe安装**

&emsp;&emsp;<font size=4>在cmd中输入pip install LabelMe开始安装。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/13c38eef42dd49daa38e2fea8dc4c22d3a376438ca8f4db3b8895867c97181d0)<br><br>
&emsp;&emsp;<font size=4>等待安装，当出现Successfully.....时，即安装成功。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/c0ecd707f7f443ba8d46479fb63a865886a5ef0b9d124e079342656c699c503e)

### **2.2.2、启动LabelMe**

&emsp;&emsp;<font size=4>在cmd输入LabelMe即可启动。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/29b9e077e47e4dfb9b6e24c7c88ce981d94e96650c064298a4641c2513b8343e)

## **2.3、使用LabelMe进行标注**

&emsp;&emsp;<font size=4>①选择原图片所在的文件夹。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/1159c2c2e8304d6090876f9f1a5030be64de9459d3154d6e8887a596592790f5)<br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/40ee9ed0f6914a27a15859001a095f22e496113f43734a478cbf3d83d3a65cc7)<br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/720a955e6d8f4098bada174227142c38c08e33ad486b4d2794df3c53e8921c0d)<br><br>
&emsp;&emsp;<font size=4>②点击`Create Polygons`，沿着目标的边缘画多边形，完成后输入目标的类别。在标注过程中，如果某个点画错了，可以按撤销快捷键可撤销该点。Windows下的撤销快捷键为`ctrl+Z`，标注完后在方框里输入该类型对应的label。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/e4d94e1e314142fc89cb69725fd2b530d45a14e46d624e49a4397b8771962c45)<br><br>
&emsp;&emsp;<font size=4>③当发现标注的类别错误时，可以点击`Edit Polygons`，然后按照该顺序依次点击，可以修改类别。(如该例中将该类别从`optic_disc`改成了`kkk`)</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/cc7131270aba43df8f7c318db1fad0e90066c0ee6f5740438f35bbd96a3ac570)<br><br>
&emsp;&emsp;<font size=4>④图片中所有目标的标注都完成后，点击`Save`保存json文件，**请将json文件和图片放在同一个文件夹里**，点击`Next Image`标注下一张图片。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/bb11d421eb3643e78c6016dc99f950fecd9db1c3ae714233ac9f98f308fa09ee)

## **2.4、一键生成数据集**

&emsp;&emsp;<font size=4>①官方代码PaddleSeg文件夹下的`./pdseg/tools`中提供的数据转换脚本`labelme2seg.py`将上述标注工具产出的数据格式转换为模型训练时所需的数据格式。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/5d15798280674bce94537a371ce3109a42fdc83b3fa74fdcb2b43987608f569a)<br><br>
&emsp;&emsp;<font size=4>②在cmd中输入`python labelme2seg.py <标注数据所在文件夹>`即可一键转换。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/221f935d66324a6982f65e64446afb1b89eb7b1283c744e6bef14384a8d65ded)<br><br>
&emsp;&emsp;<font size=4>③转换结果。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/ab248b320aec49fdb1ae187da7e60b78e2975b43054849859b435d13bbe93687)

## **2.5、可能的一些error**

&emsp;&emsp;<font size=4>①error1：</font><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/8bc65fb348fd43329afedb3c81399387317474939f57496eb4d98f877f2cc4dc)<br><br>
&emsp;&emsp;<font size=4>原因：没有将json文件和图片放在同一个文件夹里(如图所示检查)。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/68c55d7ea4934420bb87a7a814b6ddc0db050bfbf76c426ba08d12cc78263875)
