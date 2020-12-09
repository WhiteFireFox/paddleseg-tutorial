# **Windows：如何利用PaddleSeg做一个完整的项目(动态图)②**

# **PaddleSeg简介**

&emsp;&emsp;<font size=4>PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。</font><br><br>
&emsp;&emsp;<font size=4>文章将在 **“[Windows：如何利用PaddleSeg做一个完整的项目(动态图)①](https://github.com/WhiteFireFox/paddleseg-tutorial/tree/main/pdseg-dygragh/part1)”** 基础上，继续以一个例子为说明，来谈如何利用PaddleSeg完成一个项目。</font>

# **项目目录**

&emsp;&emsp;<font size=4>3、数据集准备②</font><br><br>
&emsp;&emsp;<font size=4>4、训练</font><br><br>
&emsp;&emsp;<font size=4>5、训练过程可视化</font>

# **3、数据集准备②**

&emsp;&emsp;<font size=4>注：这一部分的3.1可按照个人需求选择做或者不做，但是一定要在3.2中的处理中**对上自己图片的地址**，不然计算机不知道图片在哪里就会导致炼丹失败~</font>

## **3.1、数据集的“分门别类”**

&emsp;&emsp;<font size=4>将原图片放在origin中，将标注图片放在seg中。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/95093ff6d26e442ba5579139731bdfb06d9476c2a53c482086baaecd7b4ec532)<br><br>

## **3.2、生成train_list.txt、val_list.txt”**

&emsp;&emsp;<font size=4>train_list.txt、val_list.txt告诉了电脑图片的地址位置，让电脑知道去哪里读取你的图片。train_list.txt：训练集，val_list.txt：验证集。</font><br><br>
&emsp;&emsp;<font size=4>train_list.txt、val_list.txt文件的内容说明：</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/e1878c19cfec4efeb7d334398c99c74364075de214d9480e830f5f17fa56eb29)<br><br>
&emsp;&emsp;<font size=4>运行下列代码一键生成train_list.txt、val_list.txt：</font>


```python
import os

path_origin = 'origin/'
path_seg = 'seg/'
pic_dir = os.listdir(path_origin)

f_train = open('train_list.txt', 'w')
f_val = open('val_list.txt', 'w')

for i in range(len(pic_dir)):
    if i % 30 != 0:
        f_train.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')
    else:
        f_val.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')

f_train.close()
f_val.close()
```

# **4、训练**

&emsp;&emsp;<font size=4>在训练之前，首先了解模型的配置文件，如 **“[Windows：如何利用PaddleSeg做一个完整的项目(动态图，new)①](https://aistudio.baidu.com/aistudio/projectdetail/1123103)”** 开头所述：PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。而实现我们如何快速的完成不同模型的训练体验，就是需要依靠config文件夹下的配置文件(.yml)的切换。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/650d4719564e45ed98a5b3b12d93b908a93dc0417f574f668b27a2827c8617db)

## **4.1、按照自己需求修改配置文件**

&emsp;&emsp;<font size=4>配置文件.yaml中参数可以根据我们自己的数据情况进行设计，配置文件.yml如图所示。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/46aec63370e74f85bb84ebe3c8391aa14b174b328dc54d9497d82b8006d3610b)<br><br>
&emsp;&emsp;<font size=4>配置文件.yml可配置的参数(**[源自官网](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/dygraph/configs)**)：</font><br><br>

<h3><a id="user-content-train_dataset" class="anchor" aria-hidden="true" href="#train_dataset"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>train_dataset</h3>
<blockquote>
<p>训练数据集</p>
<ul>
<li>参数
<ul>
<li>type : 数据集类型，所支持值请参考训练配置文件</li>
<li><strong>others</strong> : 请参考对应模型训练配置文件</li>
</ul>
</li>
</ul>
</blockquote>
<hr>
<h3><a id="user-content-val_dataset" class="anchor" aria-hidden="true" href="#val_dataset"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>val_dataset</h3>
<blockquote>
<p>评估数据集</p>
<ul>
<li>参数
<ul>
<li>type : 数据集类型，所支持值请参考训练配置文件</li>
<li><strong>others</strong> : 请参考对应模型训练配置文件</li>
</ul>
</li>
</ul>
</blockquote>
<hr>
<h3><a id="user-content-batch_size" class="anchor" aria-hidden="true" href="#batch_size"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>batch_size</h3>
<blockquote>
<p>单张卡上，每步迭代训练时的数据量</p>
</blockquote>
<hr>
<h3><a id="user-content-iters" class="anchor" aria-hidden="true" href="#iters"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>iters</h3>
<blockquote>
<p>训练步数</p>
</blockquote>
<hr>
<h3><a id="user-content-optimizer" class="anchor" aria-hidden="true" href="#optimizer"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>optimizer</h3>
<blockquote>
<p>训练优化器</p>
<ul>
<li>参数
<ul>
<li>type : 优化器类型，目前只支持sgd</li>
<li>momentum : 动量</li>
<li>weight_decay : L2正则化的值</li>
</ul>
</li>
</ul>
</blockquote>
<hr>
<h3><a id="user-content-learning_rate" class="anchor" aria-hidden="true" href="#learning_rate"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>learning_rate</h3>
<blockquote>
<p>学习率</p>
<ul>
<li>参数
<ul>
<li>value : 初始学习率</li>
<li>decay : 衰减配置
<ul>
<li>type : 衰减类型，目前只支持poly</li>
<li>power : 衰减率</li>
<li>end_lr : 最终学习率</li>
</ul>
</li>
</ul>
</li>
</ul>
</blockquote>
<hr>
<h3><a id="user-content-loss" class="anchor" aria-hidden="true" href="#loss"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>loss</h3>
<blockquote>
<p>损失函数</p>
<ul>
<li>参数
<ul>
<li>types : 损失函数列表
<ul>
<li>type : 损失函数类型，目前只支持CrossEntropyLoss</li>
</ul>
</li>
<li>coef : 对应损失函数列表的系数列表</li>
</ul>
</li>
</ul>
</blockquote>
<hr>
<h3><a id="user-content-model" class="anchor" aria-hidden="true" href="#model"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>model</h3>
<blockquote>
<p>待训练模型</p>
<ul>
<li>参数
<ul>
<li>type : 模型类型，所支持值请参考模型库</li>
<li><strong>others</strong> : 请参考对应模型训练配置文件</li>
</ul>
</li>
</ul>
</blockquote>


&emsp;&emsp;<font size=4>有一点非常重要，就是train_dataset下面的type是什么东西：</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/b5222962c82c4a2a890ae109800ba845db82ee79c1584e46950ff8de1b355dd6)<br><br>
&emsp;&emsp;<font size=4>你可以认为假如是自定义数据集，那么在type这下面写Dataset来告诉网络：“我这个是自定义数据集，并且符合程序关于自定义数据的文件规范，你按照这个来读就好了~”</font><br><br>
&emsp;&emsp;<font size=4>这里官方文档给出了解释：[链接](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/dygraph/docs/data_prepare.md)</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/77ffed5b90ea4f32905e30435dd6b956080e06f3db65489da801504198b92423)

## **4.2、开始炼丹(训练)**

&emsp;&emsp;<font size=4>在cmd中输入 **python ./PaddleSeg/dygraph/train.py --config ./PaddleSeg/dygraph/configs/unet/unet.yml --do_eval --use_vdl --save_interval 5 --log_iters 1 --save_dir output**(示例) 开始训练。</font>


```python
python ./PaddleSeg/dygraph/train.py --config ./PaddleSeg/dygraph/configs/unet/unet.yml --do_eval --use_vdl --save_interval 5 --log_iters 1 --save_dir output
```

<br><br>&emsp;&emsp;<font size=4>注：请根据自己的情况调整命令行flags。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/ccea1009c54243c398bc7cb4651e3110436157a034754df7aa00122e0e49d347)<br><br>
&emsp;&emsp;<font size=4>出现如下图所示现象，可以表示为训练正常，可等待训练结束。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/423a8f5984a84a33be858f875632f1b6d47d902d22514be2bdd657fe954655bc)<br><br>
&emsp;&emsp;<font size=4>tips：命令行flags可配置的参数(**[源自官网](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/dygraph/docs/quick_start.md)**)：</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/611af9846bb844398460999f91358d81bd67fd806d7043c99cdae27d5464ccc9)

## **4.3、炼丹过程中可能的一些错误**

&emsp;&emsp;<font size=4>①当出现这个错误时候，请查看自己的配置文件.yml中是否有如图所示地加上了`num_classes`这一项。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/9caf94ae3ec04cbca183ba408e7170b429a397eea76e4d8aad97ed49a72fec49)<br><br>
&emsp;&emsp;<font size=4>②当出现这个错误时候，请查看自己的配置文件.yml中是否有如图所示地加上了`train_path`、 `val_path`这一项。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/67c7c07e347a470f9a8b46508639868ae9785102135447249fa95982183a1f36)<br><br>
&emsp;&emsp;<font size=4>③</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/fdaaa0bd9f434b6e81139c0616239f33a8204a3f17aa45fdb2f929e7c1e61429)<br><br>
&emsp;&emsp;<font size=4>解决办法：</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/85e2aff621ce4c09868a495ad7f9b5e2884a68ce536745deb6d6297201ba0f7a)<br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/f4edbea4cbf042ed888d5789a67aae83c083acdbcd8c43b9ac6779ed82fc60ea)<br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/808d512ee26c400682139dde36df646c527c33480d574cb8984e1ed762c5d334)

# **5、训练过程可视化**

&emsp;&emsp;<font size=4>通过在本地打开VisualDL实现对训练过程的可视化。</font>

## **5.1、VisualDL简介**

&emsp;&emsp;<font size=4>VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。VisualDL提供丰富的可视化功能，支持标量、图结构、数据样本可视化、直方图、PR曲线及高维数据降维呈现等诸多功能，同时VisualDL提供可视化结果保存服务，通过VDL.service生成链接，保存并分享可视化结果。</font><br><br>
&emsp;&emsp;<font size=4>**VisualDL支持浏览器种类：Chrome（81和83）、Safari 13、FireFox（77和78）、Edge（Chromium版）**</font>

## **5.2、日志logs在哪**

&emsp;&emsp;<font size=4>日志保存地址与命令行flags中的`save_dir`相关，该例中指定`output`这个文件夹作为输出地址，其输出的模型和日志log都在该目录下：</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/89e6b145a20542e3bfb3429b9278f5a186c843d925fc48b2bdd85eb613b58381)

## **5.3、启动VisualDL**

&emsp;&emsp;<font size=4>①在cmd命令行flags中加上 **--use_vdl** 。</font><br><br>


```python
python ./PaddleSeg/dygraph/train.py --config ./PaddleSeg/dygraph/configs/unet/unet.yml --do_eval --use_vdl --save_interval 5 --log_iters 1 --save_dir output
```

<br><br>&emsp;&emsp;<font size=4>②在当前目录下重开一个cmd命令行，输入：**visualdl --logdir ./output/** 。</font><br><br>


```python
visualdl --logdir ./output/
```

![](https://ai-studio-static-online.cdn.bcebos.com/3f45bc10042f4d6b90c14df134a46c732c685660d173454eb52b0d201ac502fb)<br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/b83ca862c0bc47dc8a278b7109f3647103c304fd239b4a5db4740fe2cf1bbfe2)<br><br>
&emsp;&emsp;<font size=4>③打开支持VisualDL的浏览器中，在网页链接中输入http://localhost:8040/即可。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/7d255ccc53e44c2ca3188c56b86569006de3065fd002402a91d496089648b529)<br><br>
&emsp;&emsp;<font size=4>注：打开VisualDL有点慢，请耐心等候。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/c91430b7f1694d11add2ff5008fc784ac0b2a580594a4781987f8d9ffd068900)

## **5.4、VisualDL更多信息**

&emsp;&emsp;<font size=4>**[VisualDL官网](https://github.com/PaddlePaddle/VisualDL)**</font><br><br>
