come from https://github.com/njustczr/cspdarknet53

1、DarkNet53 classification  
---------------------------
darknet53，imagenet数据集上分布式训练，模型文件（darknet53.pth）下载   
训练脚本： python main.py --dist-url env:// --dist-backend nccl --world-size 4 imagenet2012_path  
训练的时候使用了4张p40显卡，world-size设为4  
前向测试脚本： inference_darknet53.py   
百度网盘链接：https://pan.baidu.com/s/1gRzKsec0xvVZENxbnPvJmw 提取码: 99bm    
谷歌网盘链接：https://drive.google.com/file/d/1VyTXsW3O29Vr-sX5VZCpQLy_3CV4EpYX/view?usp=sharing  

2、CspDarknet53 classificaton    
-----------------------------    
cspdarknet53,imagenet数据集上分布式训练，模型文件（cspdarknet53.pth）下载  
训练脚本： python main.py --dist-url env:// --dist-backend nccl --world-size 6 imagenet2012_path  
训练的时候使用了6张p40显卡，world-size设为6  
前向测试脚本：  inference_cspdarknet53.py   
百度网盘链接：https://pan.baidu.com/s/14ZmeICTklSV-fDJoZscjDA 提取码: 5ggr   
谷歌网盘链接：https://drive.google.com/file/d/1UcU_2tysmgMAVXPlDXEvmwwWpgsNlsJY/view?usp=sharing   
