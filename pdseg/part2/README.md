# **Windows：如何利用PaddleSeg做一个完整的项目(静态图)②**

# **PaddleSeg简介**

&emsp;&emsp;<font size=4>PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。</font><br><br>
&emsp;&emsp;<font size=4>文章将在 **“[如何利用PaddleSeg做一个完整的项目(静态图)①](https://github.com/WhiteFireFox/paddleseg-tutorial/edit/main/pdseg/part2)”** 基础上，继续以一个例子为说明，来谈如何利用PaddleSeg完成一个项目。</font>

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

&emsp;&emsp;<font size=4>在训练之前，首先了解模型的配置文件，如 **“[如何利用PaddleSeg做一个完整的项目①](https://aistudio.baidu.com/aistudio/projectdetail/1101667)”** 开头所述：PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。而实现我们如何快速的完成不同模型的训练体验，就是需要依靠config文件夹下的配置文件(.yaml)的切换。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/ec01c1e7b1da4907a7ea03e1d36d26b592846c4bbfa84fe9817cdf408bd0e8cd)

## **4.1、按照自己需求修改配置文件**

&emsp;&emsp;<font size=4>配置文件.yaml中参数可以根据我们自己的数据情况进行设计，配置文件.yaml如图所示。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/4a7cb95083f0475a99dbd709c679c34f00ce0c7600d143c8b1b63bdd3fac2fb0)<br><br>
&emsp;&emsp;<font size=4>配置文件.yaml可配置的参数(**[源自官网](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.6.0/docs/config.md)**)：</font><br><br>

	########################## 基本配置 ###########################################
	# 批处理大小
	BATCH_SIZE: 1
	# 验证时图像裁剪尺寸（宽，高）
	EVAL_CROP_SIZE: tuple()
	# 训练时图像裁剪尺寸（宽，高）
	TRAIN_CROP_SIZE: tuple()
	
	########################## 数据集配置 #########################################
	DATASET:
	# 数据主目录目录
	DATA_DIR: './dataset/cityscapes/'
	# 训练集列表
	TRAIN_FILE_LIST: './dataset/cityscapes/train.list'
	# 验证集列表
	VAL_FILE_LIST: './dataset/cityscapes/val.list'
	# 测试数据列表
	TEST_FILE_LIST: './dataset/cityscapes/test.list'
	# VisualDL 可视化的数据集
	VIS_FILE_LIST: None
	# 类别数(需包括背景类)
	NUM_CLASSES: 19
	# 输入图像类型, 支持三通道'rgb',四通道'rgba',单通道灰度图'gray'
	IMAGE_TYPE: 'rgb'
	# 输入图片的通道数
	DATA_DIM: 3
	# 数据列表分割符, 默认为空格
	SEPARATOR: ' '
	# 忽略的像素标签值, 默认为255，一般无需改动
	IGNORE_INDEX: 255
	
	########################## 模型通用配置 #######################################
	MODEL:
	# 模型名称, 已支持deeplabv3p, unet, icnet，pspnet，hrnet
	MODEL_NAME: ''
	# BatchNorm类型: bn、gn(group_norm)
	DEFAULT_NORM_TYPE: 'bn'
	# 多路损失加权值
	MULTI_LOSS_WEIGHT: [1.0]
	# DEFAULT_NORM_TYPE为gn时group数
	DEFAULT_GROUP_NUMBER: 32
	# 极小值, 防止分母除0溢出，一般无需改动
	DEFAULT_EPSILON: 1e-5
	# BatchNorm动量, 一般无需改动
	BN_MOMENTUM: 0.99
	# 是否使用FP16训练
	FP16: False
	
	########################## DeepLab模型配置 ####################################
	DEEPLAB:
	    # DeepLab backbone 配置, 可选项xception_65, mobilenetv2
	    BACKBONE: "xception_65"
	    # DeepLab output stride
	    OUTPUT_STRIDE: 16
	    # MobileNet v2 backbone scale 设置
	    DEPTH_MULTIPLIER: 1.0
	    # MobileNet v2 backbone scale 设置
	    ENCODER_WITH_ASPP: True
	    # MobileNet v2 backbone scale 设置
	    ENABLE_DECODER: True
	    # ASPP是否使用可分离卷积
	    ASPP_WITH_SEP_CONV: True
	    # 解码器是否使用可分离卷积
	    DECODER_USE_SEP_CONV: True
	
	########################## UNET模型配置 #######################################
	UNET:
	    # 上采样方式, 默认为双线性插值
	    UPSAMPLE_MODE: 'bilinear'
	
	########################## ICNET模型配置 ######################################
	ICNET:
	    # RESNET backbone scale 设置
	    DEPTH_MULTIPLIER: 0.5
	    # RESNET 层数 设置
	    LAYERS: 50
	
	########################## PSPNET模型配置 ######################################
	PSPNET:
	    # RESNET backbone scale 设置
	    DEPTH_MULTIPLIER: 1
	    # RESNET backbone 层数 设置
	    LAYERS: 50
	
	########################## HRNET模型配置 ######################################
	HRNET:
	    # HRNET STAGE2 设置
	    STAGE2:
	        NUM_MODULES: 1
	        NUM_CHANNELS: [40, 80]
	    # HRNET STAGE3 设置
	    STAGE3:
	        NUM_MODULES: 4
	        NUM_CHANNELS: [40, 80, 160]
	    # HRNET STAGE4 设置
	    STAGE4:
	        NUM_MODULES: 3
	        NUM_CHANNELS: [40, 80, 160, 320]
	
	########################### 训练配置 ##########################################
	TRAIN:
	# 模型保存路径
	MODEL_SAVE_DIR: ''
	# 预训练模型路径
	PRETRAINED_MODEL_DIR: ''
	# 是否resume，继续训练
	RESUME_MODEL_DIR: ''
	# 是否使用多卡间同步BatchNorm均值和方差
	SYNC_BATCH_NORM: False
	# 模型参数保存的epoch间隔数，可用来继续训练中断的模型
	SNAPSHOT_EPOCH: 10
	
	########################### 模型优化相关配置 ##################################
	SOLVER:
	# 初始学习率
	LR: 0.1
	# 学习率下降方法, 支持poly piecewise cosine 三种
	LR_POLICY: "poly"
	# 优化算法, 支持SGD和Adam两种算法
	OPTIMIZER: "sgd"
	# 动量参数
	MOMENTUM: 0.9
	# 二阶矩估计的指数衰减率
	MOMENTUM2: 0.999
	# 学习率Poly下降指数
	POWER: 0.9
	# step下降指数
	GAMMA: 0.1
	# step下降间隔
	DECAY_EPOCH: [10, 20]
	# 学习率权重衰减，0-1
	WEIGHT_DECAY: 0.00004
	# 训练开始epoch数，默认为1
	BEGIN_EPOCH: 1
	# 训练epoch数，正整数
	NUM_EPOCHS: 30
	# loss的选择，支持softmax_loss, bce_loss, dice_loss
	LOSS: ["softmax_loss"]
	# 是否开启warmup学习策略 
	LR_WARMUP: False 
	# warmup的迭代次数
	LR_WARMUP_STEPS: 2000 
	
	########################## 测试配置 ###########################################
	TEST:
	# 测试模型路径
	TEST_MODEL: ''
	
	########################### 数据增强配置 ######################################
	AUG:
	# 图像resize的方式有三种：
	# unpadding（固定尺寸），stepscaling（按比例resize），rangescaling（长边对齐）
	AUG_METHOD: 'unpadding'
	
	# 图像resize的固定尺寸（宽，高），非负
	FIX_RESIZE_SIZE: (500, 500)
	
	# 图像resize方式为stepscaling，resize最小尺度，非负
	MIN_SCALE_FACTOR: 0.5
	# 图像resize方式为stepscaling，resize最大尺度，不小于MIN_SCALE_FACTOR
	MAX_SCALE_FACTOR: 2.0
	# 图像resize方式为stepscaling，resize尺度范围间隔，非负
	SCALE_STEP_SIZE: 0.25
	
	# 图像resize方式为rangescaling，训练时长边resize的范围最小值，非负
	MIN_RESIZE_VALUE: 400
	# 图像resize方式为rangescaling，训练时长边resize的范围最大值，
	# 不小于MIN_RESIZE_VALUE
	MAX_RESIZE_VALUE: 600
	# 图像resize方式为rangescaling, 测试验证可视化模式下长边resize的长度，
	# 在MIN_RESIZE_VALUE到MAX_RESIZE_VALUE范围内
	INF_RESIZE_VALUE: 500
	
	# 图像镜像左右翻转
	MIRROR: True
	# 图像上下翻转开关，True/False
	FLIP: False
	# 图像启动上下翻转的概率，0-1
	FLIP_RATIO: 0.5
	
	RICH_CROP:
	    # RichCrop数据增广开关，用于提升模型鲁棒性
	    ENABLE: False
	    # 图像旋转最大角度，0-90
	    MAX_ROTATION: 15
	    # 裁取图像与原始图像面积比，0-1
	    MIN_AREA_RATIO: 0.5
	    # 裁取图像宽高比范围，非负
	    ASPECT_RATIO: 0.33
	    # 亮度调节范围，0-1
	    BRIGHTNESS_JITTER_RATIO: 0.5
	    # 饱和度调节范围，0-1
	    SATURATION_JITTER_RATIO: 0.5
	    # 对比度调节范围，0-1
	    CONTRAST_JITTER_RATIO: 0.5
	    # 图像模糊开关，True/False
	    BLUR: False
	    # 图像启动模糊百分比，0-1
	    BLUR_RATIO: 0.1
	
	########################## 预测部署模型配置 ###################################
	FREEZE:
	# 预测保存的模型名称
	MODEL_FILENAME: '__model__'
	# 预测保存的参数名称
	PARAMS_FILENAME: '__params__'
	# 预测模型参数保存的路径
	SAVE_DIR: 'freeze_model'
	
	########################## 数据载入配置 #######################################
	DATALOADER:
	# 数据载入时的并发数, 建议值8
	NUM_WORKERS: 8
	# 数据载入时缓存队列大小, 建议值256
	BUF_SIZE: 256

## **4.2、开始炼丹(训练)**

&emsp;&emsp;<font size=4>在cmd中输入 **python -u ./PaddleSeg-release-v0.6.0/pdseg/train.py --cfg ./PaddleSeg-release-v0.6.0/configs/unet_optic.yaml**(示例) 开始训练。</font><br><br>
&emsp;&emsp;<font size=4>注：请根据自己的情况调整命令行flags。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/ade333c53e114a6bb83191b702a58dc3327088073e5f46168ef64968bf9f4f92)<br><br>
&emsp;&emsp;<font size=4>出现如下图所示现象，可以表示为训练正常，可等待训练结束。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/667242ede2a64cb8b22f043e2e8b31dfd9e1ed877cf14427aa0121c472a4ddd2)<br><br>
&emsp;&emsp;<font size=4>tips：命令行flags可配置的参数(**[源自官网](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.6.0/docs/config.md)**)：</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/ba91bda4ed2d489c92adc1dff9f45050c14cbb070bca45d3a1d834d45e642376)<br><br>
&emsp;&emsp;<font size=4>命令行flags配置示例：</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/2a6cf5c01574481ebc746a2501db91ea5a35fa34c21f4b46af94f29d318a1ff2)

# **5、训练过程可视化**

&emsp;&emsp;<font size=4>通过在本地打开VisualDL实现对训练过程的可视化。</font>

## **5.1、VisualDL简介**

&emsp;&emsp;<font size=4>VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。VisualDL提供丰富的可视化功能，支持标量、图结构、数据样本可视化、直方图、PR曲线及高维数据降维呈现等诸多功能，同时VisualDL提供可视化结果保存服务，通过VDL.service生成链接，保存并分享可视化结果。</font><br><br>
&emsp;&emsp;<font size=4>**VisualDL支持浏览器种类：Chrome（81和83）、Safari 13、FireFox（77和78）、Edge（Chromium版）**</font>

## **5.2、启动VisualDL**

&emsp;&emsp;<font size=4>①在cmd命令行flags中加上 **--use_vdl** 、 **--vdl_log_dir ./logs/** 。</font><br><br>
&emsp;&emsp;![](https://ai-studio-static-online.cdn.bcebos.com/1c867ca0ee4e41449ab4580151237703cf135487266b41bab8b4febb2fb3f82d)<br><br>
&emsp;&emsp;<font size=4>②在当前目录下重开一个cmd命令行，输入：**visualdl --logdir ./logs/** 。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/0d72aa66cf504741a7a5f6d80cf6b1cb867d9711dde64bae8a9d40141feefaed)<br><br>
&emsp;&emsp;<font size=4>③打开支持VisualDL的浏览器中，在网页链接中输入http://localhost:8040/即可。</font><br><br>
&emsp;&emsp;<font size=4>注：打开VisualDL有点慢，请耐心等候。</font><br><br>
![](https://ai-studio-static-online.cdn.bcebos.com/b4933fcbdcc9475991072c8b39f49ed46b46060371e74063b9f9301f7d34e922)

## **5.3、VisualDL更多信息**

&emsp;&emsp;<font size=4>**[VisualDL官网](https://github.com/PaddlePaddle/VisualDL)**</font><br><br>
