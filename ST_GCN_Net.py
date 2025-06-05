#准备环境
import os
import paddle
import Data_Processor_keypoint
from Data_Processor_keypoint import SkeletonDataset
# 添加到模型训练之前以启用自动混合精度
paddle.amp.auto_cast(enable=True)
#全局变量
BATCH_SIZE=8
#Train=1
Train =0
#GCN网络结构
class GCN(paddle.nn.Layer):
    def __init__(self,in_channels,out_channels,stride=1):
        super(GCN, self).__init__()
        self.conv1=paddle.nn.Conv2D(in_channels=in_channels,out_channels=3*out_channels,kernel_size=1,stride=1)
        #这段的意思是第一层卷积，输入通道数为in_channels，输出通道数为3*out_channels（把周围三个关节点也考虑进去，以便于计算，卷积核大小为1*1
        self.conv2=paddle.nn.Conv2D(in_channels=25*3,out_channels=25,kernel_size=1)

    def forward(self, x):
        # X----[N,C,T,V]
        # N:batch_size,C:channels,T:frames,V:joints
        x = self.conv1(x)
        N, C, T, V = paddle.shape(x)# 数据集读取器返回形状：[C=3, T=1500, V=25]
        x = paddle.reshape(x, [N, C // 3, 3, T, V])
        x = paddle.transpose(x, perm=[0, 1, 2, 4, 3])
        x = paddle.reshape(x, [N, C // 3, 3 * V, T])
        x = paddle.transpose(x, perm=[0, 2, 1, 3])
        x = self.conv2(x)
        x = paddle.transpose(x, perm=[0, 2, 1, 3])
        x = paddle.transpose(x, perm=[0, 1, 3, 2])
        return x


#TCN网络结构
class TCN(paddle.nn.Layer):
    def __init__(self,in_channels,out_channels,stride=1):
        super(TCN, self).__init__()
        self.conv=paddle.nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=(9,1),padding=(4,0),stride=(stride,1))#补零，4
        #C=64
    def forward(self, x):
        x=self.conv(x)
        return x

#ST-GCN网络结构,包含GCN和TCN，通过调用类GCN和TCN实现
class LoopNet(paddle.nn.Layer):
    def __init__(self,in_channels,out_channels,stride=1,if_res=1):
        super(LoopNet, self).__init__()
        self.bn_res=paddle.nn.BatchNorm2D(out_channels)
        self.conv_res=paddle.nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=(stride,1))
        self.gcn=GCN(in_channels=in_channels,out_channels=out_channels)
        self.bn1=paddle.nn.BatchNorm2D(out_channels)
        self.tcn=TCN(in_channels=out_channels,out_channels=out_channels,stride=stride)
        self.bn2=paddle.nn.BatchNorm2D(out_channels)
        self.if_res=if_res
        self.out_channels=out_channels
    def forward(self, x):
        if(self.if_res):#残差层
            y=self.conv_res(x)
            y=self.bn_res(y)
        x=self.gcn(x)   #gcn层
        x=self.bn1(x)   #bn层
        x=paddle.nn.functional.relu(x)
        x=self.tcn(x)
        x=self.bn2(x)
        out=x
        if(self.if_res):
            out=out+y
        out=paddle.nn.functional.relu(out)
        return out
#进一步封装网络结构
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.dropout = paddle.nn.Dropout(p=0.5)  # 增加dropout
        self.loopnet=paddle.nn.Sequential(
            LoopNet(in_channels=2,out_channels=64,if_res=0),
            LoopNet(in_channels=64,out_channels=64),
            LoopNet(in_channels=64,out_channels=64),
            LoopNet(in_channels=64,out_channels=64),
            LoopNet(in_channels=64,out_channels=128,stride=2),
            LoopNet(in_channels=128,out_channels=128),
            LoopNet(in_channels=128,out_channels=128),
            LoopNet(in_channels=128,out_channels=256,stride=2),
            LoopNet(in_channels=256,out_channels=256),
            LoopNet(in_channels=256,out_channels=256)
        )
        self.globalpooling=paddle.nn.AdaptiveAvgPool2D(output_size=(1,1))
        self.flatten=paddle.nn.Flatten()
        self.fc=paddle.nn.Linear(in_features=256,out_features=13)   #更改分类数目时只需更改out_features
    def forward(self, x):
        x=self.loopnet(x)#循环网络
        x=self.globalpooling(x)#全局池化
        x=self.flatten(x)#展平
        x = self.dropout(x)
        x=self.fc(x)#全连接用于输出概率分类
        return x
##################################训    练    部    分##########################################################
# 封装模型为一个 model 实例，便于进行后续的训练、评估和推理
MyNet_model=MyNet()#建立实例
model = paddle.Model(MyNet_model)
# 配置模型的优化器、损失函数和评估指标
#momentum = paddle.optimizer.Momentum(learning_rate=0.001, parameters=model.parameters(),use_nesterov=True, weight_decay=1e-4)

# 替换现有momentum优化器配置
# 原配置：
# momentum = paddle.optimizer.Momentum(
#     learning_rate=0.001,
#     parameters=model.parameters(),
#     use_nesterov=True,
#     weight_decay=1e-4
# )

# 新配置：LAMB优化器 + 余弦退火
# LAMB优化器配置（保持网络结构不变）
# 修改后的LAMB优化器配置（关键修正）
# optimizer = paddle.optimizer.Lamb(
#     learning_rate=0.001,
#     lamb_weight_decay=0.01,
#     beta1=0.9,
#     beta2=0.999,
#     parameters=model.parameters(),
#     grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=2.0),
#     exclude_from_weight_decay_fn=lambda param: 'batch_norm' in param.name,  # 仅检查参数名称
#     epsilon=1e-6
# )
from paddle.optimizer.lr import CosineAnnealingDecay, LinearWarmup

# 先定义基础学习率调度器
base_lr = 0.001
total_epochs = 30
# 假设每个epoch有39个step（根据您的训练日志）
total_steps = total_epochs * 39  # 30*39=1170
warmup_steps = 5 * 39  # 前5个epoch作为warmup

# 创建余弦衰减调度器
cosine_scheduler = CosineAnnealingDecay(
    learning_rate=base_lr,
    T_max=total_steps - warmup_steps  # 衰减总步数
)

# 添加线性warmup
lr_scheduler = LinearWarmup(
    learning_rate=cosine_scheduler,
    warmup_steps=warmup_steps,
    start_lr=base_lr * 0.01,  # warmup起始学习率(1%的base_lr)
    end_lr=base_lr            # warmup结束学习率
)

# 优化器配置
optimizer = paddle.optimizer.Lamb(
    learning_rate=lr_scheduler,
    lamb_weight_decay=0.003,
    beta1=0.9,
    beta2=0.999,
    parameters=model.parameters(),
    grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
    exclude_from_weight_decay_fn=lambda param: 'batch_norm' in param.name,
    epsilon=1e-6
)




#optimizer:优化器，loss:损失函数，metrics:评估指标(这里用分类准确度指标)，amp_configs:自动混合精度训练的配置

#查看模型结构
# 定义输入特征维度 [批大小, 通道数, 时间步, 关节点数]
input_shape =  (16,2,50,25) # 假设：16个样本，2个坐标特征，150帧，25关节点
#输入的维度是(32,2,300,25)

# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(optimizer=optimizer,
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

paddle.summary(
    net=MyNet(),               # 模型实例
    input_size=input_shape,    # 输入张量形状
    dtypes='float32'           # 输入数据类型
)
# 加载模型参数和优化器参数
#from STGCN_Predict import config  # 导入配置对象
# model.load("models/stgcn_model")
# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
if Train==1:
    model.fit(Data_Processor_keypoint.dataset, epochs=30, batch_size=BATCH_SIZE, verbose=1)
# 用 evaluate 在测试集上对模型进行验证
#eval_result = model.evaluate(train_custom_dataset, verbose=1)
#print(eval_result)#输出评估结果
# 直接保存模型参数（最简版）
if Train==1:
    model.save(os.path.join("/home/aistudio/output", "stgcn_model"))


#model.save("inference_model", False)  # 保存推理模型,这里后面会改成静态图模型，现在先保存动态图模型


#如果遇到兼容问题，参考https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/load_old_format_model_cn.html

