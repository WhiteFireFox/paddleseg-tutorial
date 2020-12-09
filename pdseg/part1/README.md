# **Windows：如何利用PaddleSeg做一个完整的项目(静态图)①**

# **PaddleSeg简介**

&emsp;&emsp;<font size=4>PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。</font><br><br>
&emsp;&emsp;<font size=4>文章将以一个例子为说明，来谈如何利用PaddleSeg完成一个项目。</font>

# **项目目录**

&emsp;&emsp;<font size=4>1、环境部署</font><br><br>
&emsp;&emsp;<font size=4>2、数据集准备①</font><br><br>

# **1、环境部署**

## **1.1、安装paddle**

&emsp;&emsp;<font size=4>①百度搜索paddle，并点击进入paddle官网。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/1aba3e2e34ba40809c7d340ae38616b558c567282d6d4ac5963f0c96c6b83991)<br><br>
&emsp;&emsp;<font size=4>②按照自己的电脑的配置去选择。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/4a51bd07591445cab3d430771da45a22948232501de54c2b9e9fdd9e0a867584)<br><br>
&emsp;&emsp;<font size=4>③网页会自动根据您的选择去显示相应的安装的命令，将安装的命令复制到cmd中，摁下回车，进行安装。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/cfa02eb5b1e14ce295bccdbfeb0959f9dad153443a4a4f72bc23deefe222c575)<br><br>
&emsp;&emsp;<font size=4>等待安装，当出现Successfully.....时，即安装成功。</font>

## **1.2、下载PaddleSeg**

&emsp;&emsp;<font size=4>本文提供三种下载方案。</font>

### **1.2.1、从Gitee官网直接下载(国内，较快)**

&emsp;&emsp;<font size=4>官网地址：[链接](https://gitee.com/paddlepaddle/PaddleSeg)</font><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/06838f36e0be428f9b9608f7458d45ea40e45e5f58d24f61b595c943d9beae14)

### **1.2.2、从GitHub官网直接下载(较慢)**

&emsp;&emsp;<font size=4>官网地址：[链接](https://github.com/paddlepaddle/PaddleSeg)</font><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/05d371727a654d3c9f7715da3ce63eab85bdb334ad46454f85da5e9c1aaa8333)

### **1.2.3、用git clone下载(较慢)**

&emsp;&emsp;<font size=4>使用“git clone https://github.com/PaddlePaddle/PaddleSeg.git”下载</font>

## **1.3、安装其它依赖库**

&emsp;&emsp;<font size=4>将1.2中下载下来的.zip解压得到下面这个文件夹。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/edee10bbf6e94cd284c29b985ccf66532a511b3afb994b608dabdc1b2422b2bf)<br><br>
&emsp;&emsp;<font size=4>依赖库文档在这个文件夹的requirements.txt中给出，可使用pip install -r requirements.txt直接安装。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/25ff1c545c4242eb8365d4d7711cf51958df24b14c5f4a9bb77c46790d90007e)<br><br>
&emsp;&emsp;<font size=4>等待安装，当出现Successfully.....时，即安装成功。</font><br><br>&emsp;&emsp;<font size=4>等待安装，当出现Successfully.....时，即安装成功。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/f86b58be774a493f86ce332b729feaced33d9892bf36455eac2e488c773d9c28)
![](https://ai-studio-static-online.cdn.bcebos.com/2d35178585f04d97b125150460dc51e79b315f6f077e409cb2644f12c9f277e5)

&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/7a875697b55f4de19e7ed1618f2b43d2ff6a3327a91b4cf88d2841eab2e13b7b)

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
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/a6542d60223343ef88aa0fc8e9f624a89f531d1ceadd43938455663440d93abd)<br><br>
&emsp;&emsp;<font size=4>②在cmd中输入`python labelme2seg.py <标注数据所在文件夹>`即可一键转换。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/7098f338ca3b4ebb9cfff04d1807cbf8cfd7921d53bf452f95f635e44393b057)<br><br>
&emsp;&emsp;<font size=4>③转换结果。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/ab248b320aec49fdb1ae187da7e60b78e2975b43054849859b435d13bbe93687)

## **2.5、可能的一些error**

&emsp;&emsp;<font size=4>①error1：</font><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/8bc65fb348fd43329afedb3c81399387317474939f57496eb4d98f877f2cc4dc)<br><br>
&emsp;&emsp;<font size=4>原因：没有将json文件和图片放在同一个文件夹里(如图所示检查)。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/68c55d7ea4934420bb87a7a814b6ddc0db050bfbf76c426ba08d12cc78263875)
