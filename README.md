#### 针对[《A Brain-inspired Embodied Intelligence for Fluid and Fast Reflexive Robotics Control》](https://arxiv.org/abs/2601.14628)论文的解读
<br><br><br><br><br><br><br><br><br><br>

# 皮层模块——VLM子模块

## 1、嵌入向量

在当前时刻 ![](https://latex.codecogs.com/svg.latex?t) ，机器人接收到图像 ![](https://latex.codecogs.com/svg.latex?I_t\in\mathbb{R}^{H\times{}W\times{}3}) 和来自人类的自然语言指令 ![](https://latex.codecogs.com/svg.latex?L_t\in\mathbb{R}^{\text{len}\times\text{voc}}) ，机器人皮层模块（Cortical Module）的VLM（Vision-Language Model，视觉-语言模型）子模块将对它们进行处理。<br>
图像和语言首先经过词元（token）化以及词嵌入（embedding）化，从而得到一个形状为 ![](https://latex.codecogs.com/svg.latex?\left(S,D_{\text{embedding}}\right)) 的嵌入向量（embedding vector）。这个嵌入向量包含了图像 ![](https://latex.codecogs.com/svg.latex?I_t\in\mathbb{R}^{H\times{}W\times{}3}) 和语言 ![](https://latex.codecogs.com/svg.latex?L_t\in\mathbb{R}^{\text{len}\times\text{voc}}) 的原始信息，表征了一个由图像数据和语言数据组成的序列（sequence）。

## 2、隐藏状态 ![](https://latex.codecogs.com/svg.latex?h_t\in\mathbb{R}^{S\times{}D})

VLM采用Transformer架构，其编码器（encoder）有 ![](https://latex.codecogs.com/svg.latex?N) 个隐藏层（hidden layers），每层内部都会进行自注意力（Self Attention）计算以及非线性前馈网络（Feed-Forward Network，FFN）计算，一般还会经过一次残差跳跃连接（Residual Skip Connection，深层神经网络的标准操作），最终输出一个隐藏状态 ![](https://latex.codecogs.com/svg.latex?h_t\in\mathbb{R}^{S\times{}D}) 。<br>
对第 ![](https://latex.codecogs.com/svg.latex?1) 个隐藏层而言，那个形状为 ![](https://latex.codecogs.com/svg.latex?\left(S,D_{\text{embedding}}\right)) 的嵌入向量就是它的输入，它会输出隐藏状态 ![](https://latex.codecogs.com/svg.latex?h_t^{(1)}\in\mathbb{R}^{S\times{}D}) ；<br>
对第 ![](https://latex.codecogs.com/svg.latex?2) 至 ![](https://latex.codecogs.com/svg.latex?N) 个隐藏层而言，低一层输出的隐藏状态将作为高一层的输入。

### 2.1、第 ![](https://latex.codecogs.com/svg.latex?1) 个隐藏层

第 ![](https://latex.codecogs.com/svg.latex?1) 个隐藏层会对包含原始信息的嵌入向量进行首次语义提取（通过自注意力计算以及非线性前馈网络计算实现），并输出隐藏状态 ![](https://latex.codecogs.com/svg.latex?h_t^{(1)}\in\mathbb{R}^{S\times{}D}) 。这个隐藏状态就包含了嵌入向量的浅层次语义信息，并且将被输入到第2个隐藏层。

### 2.2、第 ![](https://latex.codecogs.com/svg.latex?i) 个隐藏层（ ![](https://latex.codecogs.com/svg.latex?i\in\{2,\cdots,N\}) ）

第 ![](https://latex.codecogs.com/svg.latex?i) 个隐藏层（ ![](https://latex.codecogs.com/svg.latex?i\in\{2,\cdots,N\}) ）会对低一层输出的隐藏状态 ![](https://latex.codecogs.com/svg.latex?h_t^{(i-1)}\in\mathbb{R}^{S\times{}D}) 进行更深层次的语义提取（同样通过自注意力计算以及非线性前馈网络计算实现），并输出隐藏状态 ![](https://latex.codecogs.com/svg.latex?h_t^{(i)}\in\mathbb{R}^{S\times{}D}) 。这个隐藏状态可视为对低一层隐藏状态 ![](https://latex.codecogs.com/svg.latex?h_t^{(i-1)}\in\mathbb{R}^{S\times{}D}) 的“精炼”，它包含了嵌入向量的更深层次（更抽象）的语义信息，并且将被输入到第 ![](https://latex.codecogs.com/svg.latex?i+1) 个隐藏层。<br>
语义信息的层次深浅是相对的。低层输出的隐藏状态所包含的语义信息的层次相对较浅（例如空间细节），高层输出的隐藏状态所包含的语义信息的层次相对较深（例如物体的属性、用途这类抽象语义）。

## 3、隐藏状态栈 ![](https://latex.codecogs.com/svg.latex?\mathcal{H}_t)

总之，VLM子模块整体的输出结果可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{H}_t=\text{VLM}\left(I_t,L_t\right)=\{h_t^{(1)},\cdots,h_t^{(N)}\},">
</p>

它封装了从浅到深各种层次尺度的语义信息，表征了场景几何（低层次）、物体语义（高层次）、语言理解（高层次）等。
<br><br><br><br><br><br><br><br><br><br>

# 皮层模块——Q-Former子模块

## 1、拼接

从 ![](https://latex.codecogs.com/svg.latex?\mathcal{H}_t) 当中指定第 ![](https://latex.codecogs.com/svg.latex?l_{\text{start}}) 层到第 ![](https://latex.codecogs.com/svg.latex?l_{\text{end}}) 层的隐藏状态，总共 ![](https://latex.codecogs.com/svg.latex?M=l_{\text{end}}-l_{\text{start}}+1) 个，将它们沿行方向拼接（concatenate），得到

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{Concatenate}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right)\in\mathbb{R}^{MS\times{}D}.">
</p>

## 2、键/值空间投影

在后文中，![](https://latex.codecogs.com/svg.latex?\text{Concatenate}(\cdot)) 简记为 ![](https://latex.codecogs.com/svg.latex?\text{Concat}(\cdot)) 。<br>
随后进行投影，得到键矩阵（Key Matrix）以及值矩阵（Value Matrix），即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}K&=\text{Concat}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right)\cdot{}W_K\in\mathbb{R}^{MS\times{}D}\\{}V&=\text{Concat}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right)\cdot{}W_V\in\mathbb{R}^{MS\times{}D}\end{align*}">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?W_K,W_V\in\mathbb{R}^{D\times{}D}) 为投影矩阵。

## 3、交叉注意力

引入一个可学习的（learnable）查询矩阵（Query Matrix），即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q\in\mathbb{R}^{K\times{}D},">
</p>

让它和

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?K\in\mathbb{R}^{MS\times{}D},\quad{}V\in\mathbb{R}^{MS\times{}D}">
</p>

进行交叉注意力（Cross Attention）计算，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{Attention}\left(Q,K,V\right)=\text{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right)\cdot{}V\in\mathbb{R}^{K\times{}D},">
</p>

这个形状为 ![](https://latex.codecogs.com/svg.latex?\left(K,D\right)) 的计算结果就包含了VLM子模块所提取的隐藏状态信息（从浅层次的空间细节到深层次的抽象语义）。<br>
应注意，隐藏状态信息原本包含于一个形状为 ![](https://latex.codecogs.com/svg.latex?\left(MS,D\right)) 的矩阵，现在则包含于一个形状为 ![](https://latex.codecogs.com/svg.latex?\left(K,D\right)) 的矩阵。机器人接收的自然语言指令的长度是随机的，这就意味着序列的长度 ![](https://latex.codecogs.com/svg.latex?S) 是随机的，可以是很小的值，也可以是很大的值。无论 ![](https://latex.codecogs.com/svg.latex?MS) 的值有多小或者多大，包含隐藏状态信息的矩阵的形状始终由 ![](https://latex.codecogs.com/svg.latex?\left(MS,D\right)) 切换为固定不变的 ![](https://latex.codecogs.com/svg.latex?\left(K,D\right)) 。

## 4、仿射

对

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{Attention}\left(Q,K,V\right)\in\mathbb{R}^{K\times{}D}">
</p>

进行仿射（affine，即“投影+偏置”），

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{semantic}}=\text{Attention}\left(Q,K,V\right)\cdot{}W_{\text{affine}}+B_{\text{affine}}\in\mathbb{R}^{K\times{}D_{\text{action}}},">
</p>

其中

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?W_{\text{affine}}\in\mathbb{R}^{D\times{}D_{\text{action}}},\quad{}B_{\text{affine}}\in\mathbb{R}^{K\times{}D_{\text{action}}},">
</p>

并且 ![](https://latex.codecogs.com/svg.latex?D_{\text{action}}) 一般远远小于 ![](https://latex.codecogs.com/svg.latex?D) 。

## 5、![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{semantic}}) 的含义

![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{semantic}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 整体表征了皮层模块所生成并发送给小脑模块的“语义性潜在意图”（Semantic Latent Intention），它有 ![](https://latex.codecogs.com/svg.latex?K) 行，其每一行的 ![](https://latex.codecogs.com/svg.latex?D_{\text{action}}) 个分量共同集中表征了这种“语义性潜在意图”的某一个具体的方面。<br>
在后文中，![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{semantic}}) 简记为 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}) 。

## 6、![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}) 的紧凑性

相比于维度大小较为庞大的

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{Concat}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right)\in\mathbb{R}^{MS\times{}D},">
</p>

维度大小明显较小的

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}">
</p>

具备紧凑性（compactness），也就是说，后者相比前者用了更短的码长（更少的矩阵元素，因为 ![](https://latex.codecogs.com/svg.latex?K\times{}D_{\text{action}}\ll{}MS\times{}D) ）对隐藏状态信息进行编码，信息密度更高，编码更紧凑。尽管两者都包含了隐藏状态信息，但只有后者才能够作为皮层模块向小脑模块输出的包含语义、意图等高层抽象信息的数据。原论文作者团队明确指出：<br>
“直接将高维、可变长度的VLM隐藏状态输入运动策略在计算上是不可行的，并且容易导致过拟合。为了弥合VLM的语言-视觉空间与机器人控制流形（robot’s control manifold）之间的维度鸿沟，我们采用了一个分层查询Transformer（Layer-wise Querying Transformer，Q-Former）。”<br>
原论文作者团队将Q-Former的实质定义为“可学习的信息瓶颈”（learnable information bottleneck）。

## 7、同时捕获低层空间细节和高层语义抽象

另外需要指出的是，（原文）“不仅仅使用最终层，而是从指定的中间层索引范围 ![](https://latex.codecogs.com/svg.latex?\left[l_{\text{start}},l_{\text{end}}\right]) 当中提取特征，以同时捕获低层空间细节和高层语义抽象”，这一点是原论文作者团队特意设计的。

## 8、总结

针对Q-Former的生物学启示，原论文作者团队总结道：<br>
“该机制反映了皮层抽象的生物原理。正如大脑运动皮层向下游回路发送紧凑的群体水平意图信号（而非原始像素数据），我们的Q-Former将VLM的庞大知识蒸馏为紧凑的、以运动为中心的表征。这一语义潜码 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}) 编码了‘做什么’（例如，抓取杯子），而对‘如何做’的具体物理过程（例如，补偿摩擦力）保持不可知（agnostic），有效地将语义规划与物理调制解耦。”<br>
所谓的“群体水平意图信号”（population-level intent signals）在Q-Former中指的是 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 的 ![](https://latex.codecogs.com/svg.latex?K) 行向量分别表征“语义性潜在意图”的某一个具体的方面，而这 ![](https://latex.codecogs.com/svg.latex?K) 个向量所组成的“群体”，则表征“群体水平”的“语义性潜在意图”。<br>
总之，如果将键/值空间投影、交叉注意力、仿射这三部分计算逻辑封装在 ![](https://latex.codecogs.com/svg.latex?\text{Q-Former}(\cdot,\cdot)) 函数内部，则Q-Former子模块整体的输出结果可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}=\text{Q-Former}\left(\text{Concat}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right),Q\right)\in\mathbb{R}^{K\times{}D_{\text{action}}}.">
</p>

<br><br><br><br><br><br><br><br>

# 小脑模块——GRU子模块

## 1、采样周期和采样频率

机器人的传感器每进行一次采样，就会得到一个反映机器人物理状态本体感觉（proprioception）的状态向量 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}\in\mathbb{R}^{D_{\text{state}}}) ，它的 ![](https://latex.codecogs.com/svg.latex?D_{\text{state}}) 个分量表征了机器人的关节角度（joint angles）、速度（velocities）、六自由度力旋量（6-DoF wrench measurements）。<br>
在后文中，![](https://latex.codecogs.com/svg.latex?D_{\text{state}}) 简记为 ![](https://latex.codecogs.com/svg.latex?D_s) 。<br>
传感器的采样周期为 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) ，即每隔 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 秒便会进行 ![](https://latex.codecogs.com/svg.latex?1) 次采样。![](https://latex.codecogs.com/svg.latex?\Delta{}t) 往往是毫秒级的，例如 ![](https://latex.codecogs.com/svg.latex?\Delta{}t=0.02s=20ms) 。<br>
传感器的采样频率为 ![](https://latex.codecogs.com/svg.latex?f) ，即 ![](https://latex.codecogs.com/svg.latex?1) 秒钟之内总共进行 ![](https://latex.codecogs.com/svg.latex?f) 次采样（例如 ![](https://latex.codecogs.com/svg.latex?f=50\mathrm{Hz}) ）。<br>
既然每隔 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 秒就会进行 ![](https://latex.codecogs.com/svg.latex?1) 次采样，那么在 ![](https://latex.codecogs.com/svg.latex?1) 秒钟之内，采样的总次数就自然是 ![](https://latex.codecogs.com/svg.latex?1/\Delta{}t) 。因此，![](https://latex.codecogs.com/svg.latex?f) 与 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 的关系为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f=\frac{1}{\Delta{}t}.">
</p>

## 2、历史状态保留窗口

设定传感器的历史状态（History State）保留窗口的长度为 ![](https://latex.codecogs.com/svg.latex?H) 。也可以称之为“状态时域/状态的时间跨度”（State Horizon）。<br>
传感器的历史状态保留窗口的物理时间长度为 ![](https://latex.codecogs.com/svg.latex?T) 。<br>
既然每隔 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 秒就会进行 ![](https://latex.codecogs.com/svg.latex?1) 次采样，而允许保留的历史采样结果数量又被设定为 ![](https://latex.codecogs.com/svg.latex?H) ，那么，传感器的历史状态保留窗口的物理时间长度 ![](https://latex.codecogs.com/svg.latex?T) 就自然是 ![](https://latex.codecogs.com/svg.latex?\Delta{}t\cdot{}H) 。总之，![](https://latex.codecogs.com/svg.latex?T) 与 ![](https://latex.codecogs.com/svg.latex?H) 的关系为 ![](https://latex.codecogs.com/svg.latex?T=\Delta{}t\cdot{}H=H/f) 。<br>
历史状态保留窗口即为 ![](https://latex.codecogs.com/svg.latex?H) 个保留的历史状态的时间序列（temporal sequence）矩阵，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{S}_{t-H+1:t}\in\mathbb{R}^{H\times{}D_s}.">
</p>

原论文中的记法为 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t-H:t}\in\mathbb{R}^{H\times{}D_s}) 。考虑到从时刻 ![](https://latex.codecogs.com/svg.latex?t-H) 到时刻 ![](https://latex.codecogs.com/svg.latex?t) 的时间跨度实际上等于 ![](https://latex.codecogs.com/svg.latex?t-\left(t-H\right)+1=H+1) ，这与矩阵行数 ![](https://latex.codecogs.com/svg.latex?H) 产生矛盾，因此，笔者选择改记为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{S}_{t-H+1:t}\in\mathbb{R}^{H\times{}D_s},">
</p>

它所覆盖的时间跨度等于 ![](https://latex.codecogs.com/svg.latex?t-\left(t-H+1\right)+1=H) ，符合矩阵行数 ![](https://latex.codecogs.com/svg.latex?H) 。<br>
此外，笔者将字母另改为大写形式 ![](https://latex.codecogs.com/svg.latex?\mathbf{S}) ，是为了对单个历史状态向量（用小写的 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}) ）与包含了多个历史状态向量的时序矩阵（用大写的 ![](https://latex.codecogs.com/svg.latex?\mathbf{S}) ）进行区分。

## 3、门控循环单元

### 3.1、初始输入量 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}})

首先声明，仅就当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 而言，GRU（Gated Recurrent Unit，门控循环单元）在历史时刻 ![](https://latex.codecogs.com/svg.latex?t-H) 的隐藏状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}}) 是已知的，属于当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 下可以直接使用的已知量，实际上就是当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 的初始输入量。<br>
强调：当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 的初始输入量是时刻 ![](https://latex.codecogs.com/svg.latex?t-H) 下就已经算好的的隐藏状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}}) 。<br>
不过，如果当前时刻为 ![](https://latex.codecogs.com/svg.latex?t=H-1) ，也就是“系统刚刚开机不久”，仅仅采集到从时刻 ![](https://latex.codecogs.com/svg.latex?t=0) 到时刻 ![](https://latex.codecogs.com/svg.latex?t=H-1) 的 ![](https://latex.codecogs.com/svg.latex?H) 条传感数据（历史状态向量），并没有已知的隐藏状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}=\mathbf{h}_{H-1-H}=\mathbf{h}_{-1}\in\mathbb{R}^{D_{\text{hidden}}}) 可供使用，则不妨将其初始化为零向量，以供后续计算，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{h}_{-1}=\mathbf{0}\in\mathbb{R}^{D_{\text{hidden}}}.">
</p>

![](https://latex.codecogs.com/svg.latex?D_{\text{hidden}}) 表示GRU的隐藏层（hidden layer）维度。在后文中，![](https://latex.codecogs.com/svg.latex?D_{\text{hidden}}) 简记为 ![](https://latex.codecogs.com/svg.latex?D_h) 。<br>
为了在本篇解读笔记中严格区分时刻 ![](https://latex.codecogs.com/svg.latex?t) 与时间步 ![](https://latex.codecogs.com/svg.latex?\uptau) 这两种不同的概念，笔者在以下对GRU的描述过程中采用 ![](https://latex.codecogs.com/svg.latex?t^\prime) 来描述时刻，而非 ![](https://latex.codecogs.com/svg.latex?\uptau) 。

### 3.2、GRU的更新门

GRU的更新门（Update Gate）计算可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{t^\prime}=\sigma\left(W_z\mathbf{s}_{t^\prime}+U_z\mathbf{h}_{t^\prime-1}\right)\in\mathbb{R}^{D_h},">
</p>

其中

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?t^\prime\in\{t-H+1,\cdots,t\}">
</p>

表示当前时刻。此外，

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?W_z\in\mathbb{R}^{D_h\times{}D_s},\quad{}\mathbf{s}_{t^\prime}\in\mathbb{R}^{D_s},\quad{}U_z\in\mathbb{R}^{D_h\times{}D_h},\quad{}\mathbf{h}_{t^\prime-1}\in\mathbb{R}^{D_h}.">
</p>

![](https://latex.codecogs.com/svg.latex?\sigma(\mathbf{x})) 是Sigmoid函数，它会对输入向量 ![](https://latex.codecogs.com/svg.latex?\mathbf{x}) 的每一个分量 ![](https://latex.codecogs.com/svg.latex?x_i) 独立地计算

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\sigma\left(x_i\right)=\frac{1}{1+e^{-x_i}},">
</p>

从而输出一个与 ![](https://latex.codecogs.com/svg.latex?\mathbf{x}) 同维的向量，这个输出向量的分量的取值范围是 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 。总之，![](https://latex.codecogs.com/svg.latex?\sigma(\cdot)) 将输入向量的所有分量均压缩至 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 范围内。

### 3.3、GRU的重置门

GRU的重置门（Reset Gate）计算可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{r}_{t^\prime}=\sigma\left(W_r\mathbf{s}_{t^\prime}+U_r\mathbf{h}_{t^\prime-1}\right)\in\mathbb{R}^{D_h},">
</p>

其中

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?W_r\in\mathbb{R}^{D_h\times{}D_s},\quad{}U_r\in\mathbb{R}^{D_h\times{}D_h}.">
</p>

### 3.4、GRU的候选隐藏状态

GRU的候选隐藏状态（Candidate Hidden State，或称“候选激活向量”，Candidate Activation）计算可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widetilde{\mathbf{h}}_{t^\prime}=\tanh\left(W_c\mathbf{s}_{t^\prime}+U_c\cdot\left(\mathbf{r}_{t^\prime}\odot\mathbf{h}_{t^\prime-1}\right)\right)\in\mathbb{R}^{D_h},">
</p>

其中

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?W_c\in\mathbb{R}^{D_h\times{}D_s},\quad{}U_c\in\mathbb{R}^{D_h\times{}D_h}.">
</p>

“ ![](https://latex.codecogs.com/svg.latex?\odot) ”表示两个同维向量之间或者两个同维矩阵之间的逐元素独立乘积，结果依然是一个同维的向量/矩阵。<br>
![](https://latex.codecogs.com/svg.latex?\tanh(\mathbf{x})) 会对输入向量 ![](https://latex.codecogs.com/svg.latex?\mathbf{x}) 的每一个分量 ![](https://latex.codecogs.com/svg.latex?x_i) 独立地计算

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\tanh\left(x_i\right)=\frac{e^{x_i}-e^{-x_i}}{e^{x_i}+e^{-x_i}},">
</p>

从而输出一个与 ![](https://latex.codecogs.com/svg.latex?\mathbf{x}) 同维的向量，这个输出向量的分量的取值范围是 ![](https://latex.codecogs.com/svg.latex?\left(-1,1\right)) 。总之，![](https://latex.codecogs.com/svg.latex?\tanh(\cdot)) 将输入向量的所有分量均压缩至 ![](https://latex.codecogs.com/svg.latex?\left(-1,1\right)) 范围内。

### 3.5、GRU的隐藏状态更新

GRU的隐藏状态更新计算可形式化描述为在前一时刻隐藏状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t^\prime-1}) 与当前时刻候选隐藏状态 ![](https://latex.codecogs.com/svg.latex?\widetilde{\mathbf{h}}_{t^\prime}) 之间进行线性插值（linear interpolation，具有平滑性，与二元性相对），即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{h}_{t^\prime}=\left(1-\mathbf{z}_{t^\prime}\right)\odot\mathbf{h}_{t^\prime-1}+\mathbf{z}_{t^\prime}\odot\widetilde{\mathbf{h}}_{t^\prime}\in\mathbb{R}^{D_h}.">
</p>

## 4、GRU的网络行为和选型优势

### 4.1、平稳运动或者规律性运动期间的网络行为

当机器人的运动在时刻 ![](https://latex.codecogs.com/svg.latex?t^\prime) 依然保持平稳或者依然具备规律性时，![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{t^\prime}\in\mathbb{R}^{D_h}) 的某些维度的分量值将接近于 ![](https://latex.codecogs.com/svg.latex?0) ，使得 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t^\prime}\in\mathbb{R}^{D_h}) 的对应维度的分量值几乎等同于 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t^\prime-1}\in\mathbb{R}^{D_h}) 的对应维度的分量值，同时几乎完全屏蔽 ![](https://latex.codecogs.com/svg.latex?\widetilde{\mathbf{h}}_{t^\prime}\in\mathbb{R}^{D_h}) 在这些维度上的影响，从而实现隐藏状态对应维度的分量值在相邻时刻之间的延续。<br>
在平稳运动或者规律性运动（例如直线行走、圆周运动等）期间，隐藏状态本就应当对运动规律进行稳定编码，使其保持必要的延续性。这就必然要求对传感器传来的高频噪声进行屏蔽，防止其对隐藏状态中针对运动规律的编码造成破坏（其后果将包括但不限于机械臂抖动情况的恶化）。

### 4.2、突发碰撞时的网络行为

当机器人的运动在时刻 ![](https://latex.codecogs.com/svg.latex?t^\prime) 突发碰撞（collision）事件时，![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{t^\prime}\in\mathbb{R}^{D_h}) 的某些维度的分量值将接近于 ![](https://latex.codecogs.com/svg.latex?1) ，使得 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t^\prime}\in\mathbb{R}^{D_h}) 的对应维度的分量值几乎等同于 ![](https://latex.codecogs.com/svg.latex?\widetilde{\mathbf{h}}_{t^\prime}\in\mathbb{R}^{D_h}) 的对应维度的分量值。与此同时，![](https://latex.codecogs.com/svg.latex?\mathbf{r}_{t^\prime}\in\mathbb{R}^{D_h}) 的对应维度的分量值将接近于 ![](https://latex.codecogs.com/svg.latex?0) ，使得 ![](https://latex.codecogs.com/svg.latex?\widetilde{\mathbf{h}}_{t^\prime}\in\mathbb{R}^{D_h}) 的对应维度的分量值几乎完全由 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t^\prime}\in\mathbb{R}^{D_s})（包含了有关碰撞本体感觉的关键信息）决定，几乎完全屏蔽 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t^\prime-1}\in\mathbb{R}^{D_h}) 的影响。<br>
在突发碰撞时，先前的隐藏状态不再反映物理实际，它们所编码的运动规律在当前是无效的，当前的隐藏状态不能基于它们进行构建。如果要让当前的隐藏状态能够反映物理实际，就必须对传感器传来的、直接反映机器人碰撞本体感觉的高频关键信息进行充分利用，让其充分影响隐藏状态中针对当前物理实际的编码，同时屏蔽过往的隐藏状态，不让其对当前隐藏状态编码造成污染性的影响。

### 4.3、选型优势

GRU的更新门和重置门很好地满足了以上两种情况下分别对网络行为的要求，使得GRU能够成为一个对机器人而言合格的神经状态估计器（neurological state estimator）。<br>
原论文作者团队明确指出了GRU的选型优势：<br>
“与静态编码器（例如MLP）不同，GRU能够捕获变化的速率（rate of change）与接触瞬态（contact transients，例如碰撞脉冲），这对于在扰动下实现稳定运动至关重要。”

## 5、信息压缩

最新时刻 ![](https://latex.codecogs.com/svg.latex?t) 下的隐藏状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 就作为GRU子模块的输出结果，它充分整合了历史状态保留窗口

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{S}_{t-H+1:t}\in\mathbb{R}^{H\times{}D_s}">
</p>

所原本包含的信息。同时，与之相比，![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 是紧凑的（compact）————有关物理状态本体感觉的信息原本由一个矩阵包含，现在则由一个元素数量更少的向量包含。这实现了信息的压缩。<br>
梳理：<br>
1、在皮层模块的Q-Former子模块当中，图像、语言的各层次语义信息原本由

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\in\mathbb{R}^{MS\times{}D}">
</p>

包含，经处理后则由元素数量少得多的

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}">
</p>

包含（转化为意图信息），实现了信息的压缩。<br>
2、在此处（小脑模块的GRU子模块），有关物理状态本体感觉的信息原本由

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{S}_{t-H+1:t}\in\mathbb{R}^{H\times{}D_s}">
</p>

包含，经处理后则由元素数量更少的单个向量 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 包含，同样实现了信息的压缩。<br>
原论文作者团队将 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 称为“紧凑的、动态的上下文向量”（compact dynamic context vector），其中“动态”一词的逻辑在于每个时刻 ![](https://latex.codecogs.com/svg.latex?t) 下都会实时产生一个 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) ，“上下文”一词即指物理状态本体感觉历史。

## 6、GRU的计算量复杂度始终保持为常数

尽管 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 是实时产生的，然而这种实时计算并不会随着时间的推移而在延迟性能上发生丝毫的恶化。这是因为，![](https://latex.codecogs.com/svg.latex?H) 的本质是“滚动时域”（Receding Horizon），它是预先设定好的超参数，它决定了机器人的小脑模块在当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 下，只会按时序依次使用传感器最近采样的 ![](https://latex.codecogs.com/svg.latex?H) 条传感数据（历史状态向量）

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t-H+1},\mathbf{s}_{t-H+2},\cdots,\mathbf{s}_{t-1},\mathbf{s}_t\in\mathbb{R}^{D_s}">
</p>

来对初始输入的隐藏状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}}) 进行固定的 ![](https://latex.codecogs.com/svg.latex?H) 轮处理，而完全不会理睬更久远的传感器传感数据（它们的信息早已被压缩到 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}}) 里面了，自然不需要理睬）。这从输入层面就直接限制了计算量（只有初始输入的 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}}) 以及依次输入的 ![](https://latex.codecogs.com/svg.latex?H) 个历史状态向量需要作为输入量参与计算），同时单轮计算流程完全固定（更新门、重置门、候选隐藏状态、隐藏状态更新）。<br>
于是，在每个时刻 ![](https://latex.codecogs.com/svg.latex?t) 下，GRU的计算量复杂度始终保持为常数，并不会随时间的推移而保持增长。

## 7、总结
总之，GRU子模块整体的输出结果可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{h}_t=\text{GRU}\left(\mathbf{h}_{t-H},\mathbf{S}_{t-H+1:t}\right)\in\mathbb{R}^{D_h}.">
</p>

<br><br><br><br><br><br><br><br>

# 小脑模块——Gated FiLM子模块

## 1、门控因子

计算一个可学习的门控因子（gating factor），即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{g}_t=\sigma\left(W_g\cdot\text{Proj}\left(\mathbf{h}_t\right)\right)\in\mathbb{R}^{D_{\text{action}}},">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?W_g\in\mathbb{R}^{D_{\text{action}}\times{}D_{\text{action}}}) 。另外，

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{Proj}\left(\mathbf{h}_t\right)">
</p>

可视为 ![](https://latex.codecogs.com/svg.latex?W_P\cdot\mathbf{h}_t) ，其中 ![](https://latex.codecogs.com/svg.latex?W_P\in\mathbb{R}^{D_{\text{action}}\times{}D_h}) 。<br>
![](https://latex.codecogs.com/svg.latex?\text{Proj}\left(\cdot\right)) 操作是必要的，它将 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t) 的维度数由 ![](https://latex.codecogs.com/svg.latex?D_h) 转换为 ![](https://latex.codecogs.com/svg.latex?D_{\text{action}}) ，从而与 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 保持维度对齐。

## 2、仿射变换参数

计算一组仿射变换参数————缩放参数 ![](https://latex.codecogs.com/svg.latex?\gamma_t) 和偏移参数 ![](https://latex.codecogs.com/svg.latex?\beta_t) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\gamma_t&=f_\gamma\left(\mathbf{h}_t\right)\in\mathbb{R}^K\\{}\beta_t&=f_\beta\left(\mathbf{h}_t\right)\in\mathbb{R}^K\end{align*}">
</p>

## 3、![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 的仿射变换

将门控因子和仿射变换参数应用于 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 的仿射变换，得到 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{modulation}}\in\mathbb{R}^K) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{modulation}}=\left(1+\gamma_t\right)\odot\left(\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t\right)+\beta_t\in\mathbb{R}^K.">
</p>

## 4、门控因子的网络行为

原论文作者团队指出，门控因子 ![](https://latex.codecogs.com/svg.latex?\mathbf{g}_t\in\mathbb{R}^{D_{\text{action}}}) 用于“防止在稳定阶段本体感觉噪声淹没语义意图”，“选择性地调节允许多少物理上下文影响皮层计划”。<br>
也就是说，不让 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h})（经必要的维度对齐后）直接影响 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) ，即不采用

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\cdot\text{Proj}\left(\mathbf{h}_t\right)\in\mathbb{R}^K,">
</p>

而是额外引入一个门控矩阵 ![](https://latex.codecogs.com/svg.latex?W_g\in\mathbb{R}^{D_{\text{action}}\times{}D_{\text{action}}}) ，让它专门对包含本体感觉信息的 ![](https://latex.codecogs.com/svg.latex?\text{Proj}\left(\mathbf{h}_t\right)\in\mathbb{R}^{D_{\text{action}}}) 进行变换（再辅以Sigmoid非线性激活函数），即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t\in\mathbb{R}^K,\quad{}\mathbf{g}_t=\sigma\left(W_g\cdot\text{Proj}\left(\mathbf{h}_t\right)\right)\in\mathbb{R}^{D_{\text{action}}}.">
</p>

针对 ![](https://latex.codecogs.com/svg.latex?\text{Proj}\left(\mathbf{h}_t\right)\in\mathbb{R}^{D_{\text{action}}}) 所包含的本体感觉信息，![](https://latex.codecogs.com/svg.latex?W_g\in\mathbb{R}^{D_{\text{action}}\times{}D_{\text{action}}}) 既可以选择显著抑制，也可以选择轻微抑制，也可以选择几乎完整保留。

### 4.1、平稳运动或者规律性运动期间的网络行为

当平稳运动或者规律性运动时，![](https://latex.codecogs.com/svg.latex?W_g\in\mathbb{R}^{D_{\text{action}}\times{}D_{\text{action}}}) 会显著抑制 ![](https://latex.codecogs.com/svg.latex?\text{Proj}\left(\mathbf{h}_t\right)\in\mathbb{R}^{D_{\text{action}}}) 所包含的高频传感噪声，使得门控因子 ![](https://latex.codecogs.com/svg.latex?\mathbf{g}_t\in\mathbb{R}^{D_{\text{action}}}) 不再包含这种噪声，从而避免在执行向量乘法运算

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t">
</p>

时，“淹没”了 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 所包含的语义意图信息。

### 4.2、突发碰撞时的网络行为

当突发碰撞时，![](https://latex.codecogs.com/svg.latex?W_g\in\mathbb{R}^{D_{\text{action}}\times{}D_{\text{action}}}) 会几乎完整保留 ![](https://latex.codecogs.com/svg.latex?\text{Proj}\left(\mathbf{h}_t\right)\in\mathbb{R}^{D_{\text{action}}}) 所包含的关键的本体感觉信息，强制门控因子 ![](https://latex.codecogs.com/svg.latex?\mathbf{g}_t\in\mathbb{R}^{D_{\text{action}}}) 包含这种关键信息，从而在执行向量乘法运算

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t">
</p>

时，门控因子 ![](https://latex.codecogs.com/svg.latex?\mathbf{g}_t\in\mathbb{R}^{D_{\text{action}}}) 能够充分修正 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 所原本包含的、已经过时的语义意图信息。

### 4.3、门控类型区分

需要指出的是，![](https://latex.codecogs.com/svg.latex?W_g\in\mathbb{R}^{D_{\text{action}}\times{}D_{\text{action}}}) 并不是逐分量乘积式门控（即Hadamard积，运算符号为“ ![](https://latex.codecogs.com/svg.latex?\odot) ”，每个分量独立地与 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 区间内的某个小数相乘，实现按比例削减），而是矩阵-向量乘法式门控（matrix-vector multiplication，也就是点积，运算符号为“ ![](https://latex.codecogs.com/svg.latex?\cdot) ”，存在信息聚合现象）。

## 5、![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 的仿射变换其实就是仿射变换参数对它进行的逐特征线性调制

![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\in\mathbb{R}^{K\times{}D_{\text{action}}}) 在经过门控调节变换为 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t\in\mathbb{R}^K) 之后，接下来，就会接受仿射变换参数 ![](https://latex.codecogs.com/svg.latex?\gamma_t,\beta_t\in\mathbb{R}^K) 的“逐特征线性调制”（Feature-wise Linear Modulation，FiLM），即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{modulation}}=\left(1+\gamma_t\right)\odot\left(\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t\right)+\beta_t\in\mathbb{R}^K.">
</p>

在后文中，![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{modulation}}) 简称为 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}) 。

### 5.1、仿射变换参数在机器人突发碰撞时的网络行为

如果传感器检测到碰撞，那么这会反映为 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 当中某些维度分量值的突变。<br>
此时，接收 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 作为输入的神经网络 ![](https://latex.codecogs.com/svg.latex?f_\gamma(\cdot)) 所输出的 ![](https://latex.codecogs.com/svg.latex?\gamma_t\in\mathbb{R}^K) 当中，某些维度的分量值将会接近于 ![](https://latex.codecogs.com/svg.latex?-1) ，而 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t\in\mathbb{R}^K) 当中的同一批维度是用于对“前进速度”进行编码的。![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}\cdot\mathbf{g}_t\in\mathbb{R}^K) 中这些维度的分量值与

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left(1+\gamma_t^{(j)}\right)\approx\left(1+(-1)\right)=0">
</p>

（ ![](https://latex.codecogs.com/svg.latex?\gamma_t^{(j)}) 表示 ![](https://latex.codecogs.com/svg.latex?\gamma_t) 的某个分量）相乘后，数值接近于 ![](https://latex.codecogs.com/svg.latex?0) ，其原本包含的信息也就被抑制了。<br>
与此同时，同样接收 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 作为输入的神经网络 ![](https://latex.codecogs.com/svg.latex?f_\beta(\cdot)) 所输出的 ![](https://latex.codecogs.com/svg.latex?\beta_t\in\mathbb{R}^K) 当中，同一批维度的分量值将表征“缩回”（retraction，例如机械臂碰撞后主动缩回），从而“有效地实时重写运动计划”（原文）。<br>
原论文作者团队明确指出：<br>
“小脑的核心计算作用是根据当前物理约束来调制皮层指令。”
<br><br><br><br><br><br><br><br><br><br>

# 小脑模块——迭代式优化循环

## 1、本体感觉预期状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}\in\mathbb{R}^{D_s})

通过某种机制，可以获得未来时刻 ![](https://latex.codecogs.com/svg.latex?t+1)（未来的第一个时刻）下的本体感觉预期状态（anticipated state）![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}\in\mathbb{R}^{D_s}) 。<br>
需要指出的是，原文中仅提到 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}) 本身，并未给出其产生的具体机制。笔者选择将其形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}=\text{Anticipate}\left(\text{unknown}\;\,\text{inputs}\right).">
</p>

## 2、![](https://latex.codecogs.com/svg.latex?\text{Refine}\left(\cdot,\cdot\right)) 模块

定义 ![](https://latex.codecogs.com/svg.latex?\text{Refine}\left(\cdot,\cdot\right)) 模块，它封装了GRU子模块和Gated FiLM子模块的计算流程。它对输入进行循环迭代，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{\left(i+1\right)}\leftarrow\text{Refine}\left(\mathbf{z}_{\text{mod}}^{(i)},\mathbf{s}_{t+1}^{(i)}\right),">
</p>

本体感觉预期状态也将同步进行循环迭代，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}^{\left(i+1\right)}\leftarrow\text{Update}\left(\mathbf{s}_{t+1}^{(i)},\text{other}\;\,\text{unknown}\;\,\text{inputs}\right).">
</p>

设定循环迭代的次数为 ![](https://latex.codecogs.com/svg.latex?I) ，以上两式中的 ![](https://latex.codecogs.com/svg.latex?i) 便满足

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?i\in\{0,1,2,\cdots,I-1\}.">
</p>

## 3、第一次迭代

以 ![](https://latex.codecogs.com/svg.latex?i=0) 为例，第一次迭代产生 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{(1)}\in\mathbb{R}^K) 的全过程

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{\left(1\right)}\leftarrow\text{Refine}\left(\mathbf{z}_{\text{mod}}^{(0)},\mathbf{s}_{t+1}^{(0)}\right)">
</p>

可具体描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\mathbf{S}_{t-H+2:t+1}^{(0)}&=\text{Concat}\left(\mathbf{S}_{t-H+2:t},\mathbf{s}_{t+1}^{(0)}\right)\\{}&=\text{Concat}\left(\mathbf{S}_{t-H+2:t},\mathbf{s}_{t+1}\right)\in\mathbb{R}^{H\times{}D_s}\\{}\mathbf{h}_{t+1}^{(0)}&=\text{GRU}\left(\mathbf{h}_{t-H+1},\mathbf{S}_{t-H+2:t+1}^{(0)}\right)\in\mathbb{R}^{D_h}\\{}\mathbf{g}_{t+1}^{(0)}&=\sigma\left(W_g\cdot\text{Proj}\left(\mathbf{h}_{t+1}^{(0)}\right)\right)\in\mathbb{R}^{D_{\text{action}}}\\{}\gamma_{t+1}^{(0)}&=f_\gamma\left(\mathbf{h}_{t+1}^{(0)}\right)\in\mathbb{R}^K,\quad{}\beta_{t+1}^{(0)}=f_\beta\left(\mathbf{h}_{t+1}^{(0)}\right)\in\mathbb{R}^K\\{}\mathbf{z}_{\text{mod}}^{(1)}&=\left(1+\gamma_{t+1}^{(0)}\right)\odot\left(\mathbf{z}_{\text{mod}}^{(0)}\cdot\mathbf{g}_{t+1}^{(0)}\right)+\beta_{t+1}^{(0)}\\{}&=\left(1+\gamma_{t+1}^{(0)}\right)\odot\left(\mathbf{z}_{\text{mod}}\cdot\mathbf{g}_{t+1}^{(0)}\right)+\beta_{t+1}^{(0)}\in\mathbb{R}^K\end{align*}">
</p>

本体感觉预期状态同步迭代，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}^{(1)}\leftarrow\text{Update}\left(\mathbf{s}_{t+1}^{(0)},\text{other}\;\,\text{unknown}\;\,\text{inputs}\right).">
</p>

应注意，第一次迭代时，![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}^{(0)}) 指的就是 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}) ，![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{(0)}) 指的就是 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}) 。

## 4、第二次迭代

以 ![](https://latex.codecogs.com/svg.latex?i=1) 为例，第二次迭代产生 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{(2)}\in\mathbb{R}^K) 的全过程

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{\left(2\right)}\leftarrow\text{Refine}\left(\mathbf{z}_{\text{mod}}^{(1)},\mathbf{s}_{t+1}^{(1)}\right)">
</p>

可具体描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\mathbf{S}_{t-H+2:t+1}^{(1)}&=\text{Concat}\left(\mathbf{S}_{t-H+2:t},\mathbf{s}_{t+1}^{(1)}\right)\in\mathbb{R}^{H\times{}D_s}\\{}\mathbf{h}_{t+1}^{(1)}&=\text{GRU}\left(\mathbf{h}_{t-H+1},\mathbf{S}_{t-H+2:t+1}^{(1)}\right)\in\mathbb{R}^{D_h}\\{}\mathbf{g}_{t+1}^{(1)}&=\sigma\left(W_g\cdot\text{Proj}\left(\mathbf{h}_{t+1}^{(1)}\right)\right)\in\mathbb{R}^{D_{\text{action}}}\\{}\gamma_{t+1}^{(1)}&=f_\gamma\left(\mathbf{h}_{t+1}^{(1)}\right)\in\mathbb{R}^K,\quad{}\beta_{t+1}^{(1)}=f_\beta\left(\mathbf{h}_{t+1}^{(1)}\right)\in\mathbb{R}^K\\{}\mathbf{z}_{\text{mod}}^{(2)}&=\left(1+\gamma_{t+1}^{(1)}\right)\odot\left(\mathbf{z}_{\text{mod}}^{(1)}\cdot\mathbf{g}_{t+1}^{(1)}\right)+\beta_{t+1}^{(1)}\in\mathbb{R}^K\end{align*}">
</p>

本体感觉预期状态同步迭代，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}^{(2)}\leftarrow\text{Update}\left(\mathbf{s}_{t+1}^{(1)},\text{other}\;\,\text{unknown}\;\,\text{inputs}\right).">
</p>

## 5、原文引用

原论文作者团队指出：<br>
“小脑功能的一个最典型的特征是其基于预期感觉反馈持续修正运动意图的能力。在我们的模型中，这一预测校正通过迭代式优化循环（Iterative Refinement Loop）来实现。”<br>
针对这种迭代式优化循环，原论文作者团队认为：<br>
“这一递归过程充当计算性的‘心智模拟’，使智能体能够通过在实际脊髓驱动之前预先校正预期的动力学误差（例如，重力或摩擦力补偿），最小化仿真到现实的差距（Sim-to-Real gap）。”<br>
针对小脑模块的生物学启示，原论文作者团队总结道：<br>
“该架构对传出副本（Efference Copy）原理进行了实例化。语义潜码 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}) 代表‘意图的’运动（传出副本，efference copy），而 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t) 代表‘实际的’感觉反馈（再传入，re-afference）。FiLM层计算差异（感觉预测误差）并施加必要的校正。这解释了为什么我们的模型表现出‘数字肌肉记忆’————即自适应地平滑轨迹并抑制扰动的能力，而无需重新调用缓慢且计算昂贵的皮层规划器。”
<br><br><br><br><br><br><br><br><br><br>

# 脊髓模块——LIF动力学

设定当前的离散时间步为 ![](https://latex.codecogs.com/svg.latex?\uptau) ，相邻的离散时间步之间的间隔（即步长）为 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) ，当前时刻为 ![](https://latex.codecogs.com/svg.latex?t=\uptau\Delta{}t) 。<br>
于是，前一个刚刚过去的时间步为 ![](https://latex.codecogs.com/svg.latex?\uptau-1) ，对应的时刻为 ![](https://latex.codecogs.com/svg.latex?\left(\uptau-1\right)\Delta{}t) 。<br>
神经元细胞膜（membrane）的脂质成分是绝缘的，而膜内、膜外则充满了导电的溶液（分别称为“细胞内液”、“细胞外液”）。神经元细胞膜内存在一个电位（potential，也称“电势”），膜外也同时存在一个电位，它们之间由于绝缘的关系而无法等同，那么就自然而然地存在一个电位差（即“电压”，voltage）。因此，神经元细胞膜的脂质成分可视为电容 ![](https://latex.codecogs.com/svg.latex?C_m) 。<br>
神经元细胞膜上镶嵌的离子通道蛋白在宏观上可视为具备导电性，并具有宏观意义上的电阻 ![](https://latex.codecogs.com/svg.latex?R_m) 。<br>
在“漏电-积分-发放”（Leaky Integrate-and-Fire，LIF）简化模型中，神经元是分层的。如果低一层的某一个神经元发放脉冲（spike），则高一层的某些神经元会各自接收到由它引起的一对一专属输入电流（input current）。如果低一层的这个神经元不发放脉冲，则高一层的这些神经元各自均不会接收到由它引起的一对一专属输入电流。<br>
应注意，实际的生理过程远远比这复杂，这仅仅是适配建模实现的必要简化。本篇解读笔记聚焦于具身智能工程实现技术NeuroVLA，不对精细的神经科学过程展开冗余论述。<br>
针对当前时刻 ![](https://latex.codecogs.com/svg.latex?\uptau\Delta{}t) 下的第 ![](https://latex.codecogs.com/svg.latex?l-1) 层第 ![](https://latex.codecogs.com/svg.latex?j) 个神经元，用

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right)\in\{0,1\}">
</p>

来指示该神经元在当前时刻是否发放脉冲。![](https://latex.codecogs.com/svg.latex?s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right)=0) 表示不发放，![](https://latex.codecogs.com/svg.latex?s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right)=1) 表示发放。<br>
由第 ![](https://latex.codecogs.com/svg.latex?l-1) 层第 ![](https://latex.codecogs.com/svg.latex?j) 个神经元引起的、并由第 ![](https://latex.codecogs.com/svg.latex?l) 层第 ![](https://latex.codecogs.com/svg.latex?i) 个神经元接收的、一对一专属的输入电流可记为 ![](https://latex.codecogs.com/svg.latex?I_{ij}) 。<br>
用

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_{ij}\cdot{}s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right)">
</p>

来表示两种情形。![](https://latex.codecogs.com/svg.latex?I_{ij}\cdot{}s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right)=0) 表示未接收，![](https://latex.codecogs.com/svg.latex?I_{ij}\cdot{}s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right)=I_{ij}) 表示接收。<br>
第 ![](https://latex.codecogs.com/svg.latex?l) 层第 ![](https://latex.codecogs.com/svg.latex?i) 个神经元有可能接收到由第 ![](https://latex.codecogs.com/svg.latex?l-1) 层全体神经元各自引起的一对一专属输入电流（可以等于 ![](https://latex.codecogs.com/svg.latex?0) ，表示“未接收”）。因此，该神经元接收的“输入电流积分”（即输入电流总和）可表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_{\text{in}}\left(\uptau\Delta{}t\right)=\sum_jI_{ij}\cdot{}s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right).">
</p>

它就是“Leaky Integrate-and-Fire”当中“Integrate”（积分）一词的实质。<br>
![](https://latex.codecogs.com/svg.latex?I_{\text{in}}\left(\uptau\Delta{}t\right)) 存在两种去向：<br>
1、通过电阻 ![](https://latex.codecogs.com/svg.latex?R_m) “泄漏”出去。这体现“Leaky”一词的实质。<br>
2、流向膜内，使膜内的正电荷增多，膜内电位升高，从而在膜外电位基本保持恒定的同时，使膜内外电位差增大。这本质上就是对电容 ![](https://latex.codecogs.com/svg.latex?C_m) 进行“充电”。<br>
由此，可以建立一个局部电路模型：<br>
![](https://latex.codecogs.com/svg.latex?I_{\text{in}}\left(\uptau\Delta{}t\right)) 输入节点，并联的 ![](https://latex.codecogs.com/svg.latex?R_m,C_m) 分别在“电阻支路”、“电容支路”上，电阻支路电流 ![](https://latex.codecogs.com/svg.latex?I_{R_m}\left(\uptau\Delta{}t\right)) 、电容支路电流 ![](https://latex.codecogs.com/svg.latex?I_{C_m}\left(\uptau\Delta{}t\right)) 均从节点输出。<br>
![](https://latex.codecogs.com/svg.latex?I_{\text{in}}\left(\uptau\Delta{}t\right),I_{R_m}\left(\uptau\Delta{}t\right),I_{C_m}\left(\uptau\Delta{}t\right)) 的关系式即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_{\text{in}}\left(\uptau\Delta{}t\right)=I_{R_m}\left(\uptau\Delta{}t\right)+I_{C_m}\left(\uptau\Delta{}t\right).">
</p>

对于普适情形下的电流 ![](https://latex.codecogs.com/svg.latex?I) 、电压（即电位差）![](https://latex.codecogs.com/svg.latex?U) 、电阻 ![](https://latex.codecogs.com/svg.latex?R) ，存在关系式

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I=\frac{U}{R}.">
</p>

若将“流过电阻的电流”专门记为 ![](https://latex.codecogs.com/svg.latex?I_R) ，则有

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_R=\frac{U}{R}.">
</p>

对于普适情形下的（正）电荷量 ![](https://latex.codecogs.com/svg.latex?Q) 、电容 ![](https://latex.codecogs.com/svg.latex?C) 、电压 ![](https://latex.codecogs.com/svg.latex?U) ，存在关系式

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q=CU,">
</p>

从而有

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?dQ=d\left(CU\right)=C\cdot{}dU,">
</p>

从而

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{dQ}{dt}=C\cdot\frac{dU}{dt},">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?dQ/dt) 即为电流 ![](https://latex.codecogs.com/svg.latex?I) 的定义式。因此，若将“充电电流”专门记为 ![](https://latex.codecogs.com/svg.latex?I_C) ，则有

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_C=C\cdot\frac{dU}{dt}.">
</p>

综上所述，针对当前时刻 ![](https://latex.codecogs.com/svg.latex?t=\uptau\Delta{}t) 下的第 ![](https://latex.codecogs.com/svg.latex?l) 层第 ![](https://latex.codecogs.com/svg.latex?i) 个神经元，存在关系式

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\sum_jI_{ij}\cdot{}s_j^{\left(l-1\right)}\left(t\right)=\frac{u_i^{(l)}(t)}{R_m}+C_m\cdot\frac{du_i^{(l)}(t)}{dt},">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)) 表示膜内电位，同时膜外电位定义为 ![](https://latex.codecogs.com/svg.latex?0)（可以认为膜外是“接地”的），因此 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)) 还同时表示膜内、膜外的电位差（电压）。<br>
根据这个方程，可以推导出 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)) 在离散时间步情形下的前向欧拉方法近似解（迭代式）。后文将展示详细的推导过程。<br>
首先，该方程可整理为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{1}{R_mC_m}\cdot{}u_i^{(l)}(t)+\frac{du_i^{(l)}(t)}{dt}=\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t),">
</p>

方程中同时含有未知函数及其导数（即 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)) 及其导数 ![](https://latex.codecogs.com/svg.latex?du_i^{(l)}(t)/dt) ），描述了未知函数与其导数之间的关系。因此，该方程属于微分方程（Differential Equation）。<br>
该微分方程只含有一个自变量（即 ![](https://latex.codecogs.com/svg.latex?t) ）。因此，该微分方程属于常微分方程（Ordinary Differential Equation，ODE）。此类方程形如

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?F\left(x,y,y^\prime,\cdots,y^{(n)}\right)=f(x),">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?y) 是关于自变量 ![](https://latex.codecogs.com/svg.latex?x) 的未知函数，![](https://latex.codecogs.com/svg.latex?f(x)) 是关于自变量 ![](https://latex.codecogs.com/svg.latex?x) 的另一个与未知函数 ![](https://latex.codecogs.com/svg.latex?y) 无关的函数（可以是普通函数，也可以是常数，也可以是 ![](https://latex.codecogs.com/svg.latex?0) ）。当 ![](https://latex.codecogs.com/svg.latex?f(x)\neq{}0) 时，该常微分方程属于非齐次常微分方程，![](https://latex.codecogs.com/svg.latex?f(x)) 称为“非齐次项”。当 ![](https://latex.codecogs.com/svg.latex?f(x)=0) 时，该常微分方程就属于齐次常微分方程。<br>
该常微分方程中出现的最高阶导数（即 ![](https://latex.codecogs.com/svg.latex?du_i^{(l)}(t)/dt) ）的阶数为 ![](https://latex.codecogs.com/svg.latex?1) 。因此，该常微分方程属于一阶常微分方程。<br>
该常微分方程中未知函数及其各阶导数（即 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)) 以及仅有的 ![](https://latex.codecogs.com/svg.latex?du_i^{(l)}(t)/dt) ）均为一次幂。因此，该常微分方程同时属于线性常微分方程。<br>
该常微分方程的右侧含有不为 ![](https://latex.codecogs.com/svg.latex?0) 的非齐次项，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t).">
</p>

因此，该常微分方程同时属于非齐次常微分方程。<br>
前向欧拉方法（Forward Euler Method）用于常微分方程数值求解（未知函数 ![](https://latex.codecogs.com/svg.latex?y) 的函数值求解），属于显式单步法，其本质在于：<br>
用前一时间步的导数（斜率）进行“直线外推”，来估计当前时间步的函数值。在几何上，这相当于用一系列特别短小的折线（在时间轴上的投影长度仅仅等于单个离散时间步长 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) ）来逼近真实的解曲线。<br>
一般地，对于一阶线性常微分方程 ![](https://latex.codecogs.com/svg.latex?y^\prime=f(y)) ，其中的 ![](https://latex.codecogs.com/svg.latex?f(y)) 已包含可能存在的非齐次项，这些非齐次项可视为仅针对于 ![](https://latex.codecogs.com/svg.latex?y) 而言的常数。具体而言，![](https://latex.codecogs.com/svg.latex?f(y)) 可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f(y)=\alpha{}y+\beta,">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?\beta) 不为0时，它就是非齐次项（既可以是常数，也可以是关于 ![](https://latex.codecogs.com/svg.latex?x) 的另一个无关函数）。![](https://latex.codecogs.com/svg.latex?\beta) 为0时，这个一阶线性常微分方程就是齐次的。<br>
该一阶线性常微分方程在当前时刻 ![](https://latex.codecogs.com/svg.latex?t=\uptau\Delta{}t) 的前向欧拉近似解可表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?y(t)=y\left(t-\Delta{}t\right)+\Delta{}t\cdot{}f\left(y\left(t-\Delta{}t\right)\right),">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 为步长。<br>
前向欧拉方法是数值估计方法而非数值精确解析方法，产生的数值误差是结构性的。以下对前向欧拉方法的数值稳定性展开论述。<br>
对于一般的一阶线性常微分方程 ![](https://latex.codecogs.com/svg.latex?y^\prime=f(y))（其中 ![](https://latex.codecogs.com/svg.latex?f(y)=\alpha{}y+\beta) ），有

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}y(t)&=y\left(t-\Delta{}t\right)+\Delta{}t\cdot{}f\left(y\left(t-\Delta{}t\right)\right)\\&=y\left(t-\Delta{}t\right)+\Delta{}t\cdot\left[\alpha{}y\left(t-\Delta{}t\right)+\beta\right]\\&=\left(1+\alpha\Delta{}t\right)\cdot{}y\left(t-\Delta{}t\right)+\beta\Delta{}t.\end{align*}">
</p>

设定两个不同的初始值 ![](https://latex.codecogs.com/svg.latex?y(0)) 和 ![](https://latex.codecogs.com/svg.latex?\widetilde{y}(0)) ，则有

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}y(t)&=\left(1+\alpha\Delta{}t\right)\cdot{}y\left(t-\Delta{}t\right)+\beta\Delta{}t\\{}\widetilde{y}(t)&=\left(1+\alpha\Delta{}t\right)\cdot\widetilde{y}\left(t-\Delta{}t\right)+\beta\Delta{}t\end{align*}">
</p>

在时刻 ![](https://latex.codecogs.com/svg.latex?t-\Delta{}t) ，数值之差为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?d_{t-\Delta{}t}=y\left(t-\Delta{}t\right)-\widetilde{y}\left(t-\Delta{}t\right),">
</p>

而在时刻 ![](https://latex.codecogs.com/svg.latex?t) ，数值之差变化为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}d_t&=y(t)-\widetilde{y}(t)\\&=\left[\left(1+\alpha\Delta{}t\right)\cdot{}y\left(t-\Delta{}t\right)+\beta\Delta{}t\right]-\left[\left(1+\alpha\Delta{}t\right)\cdot\widetilde{y}\left(t-\Delta{}t\right)+\beta\Delta{}t\right]\\&=\left(1+\alpha\Delta{}t\right)\cdot\left[y\left(t-\Delta{}t\right)-\widetilde{y}\left(t-\Delta{}t\right)\right]\\&=\left(1+\alpha\Delta{}t\right)\cdot{}d_{t-\Delta{}t},\end{align*}">
</p>

数值的稳定性要求差值不能随时间的推移而增长，否则任意微小的初始差异都会被持续放大，导致方法不稳定，从而不可用。具体而言，对于任意时刻 ![](https://latex.codecogs.com/svg.latex?t) ，都必须满足

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left\lvert{}d_t\right\rvert\leq\left\lvert{}d_{t-\Delta{}t}\right\rvert.">
</p>

代入这个不等式，可得

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left\lvert\left(1+\alpha\Delta{}t\right)\cdot{}d_{t-\Delta{}t}\right\rvert\leq\left\lvert{}d_{t-\Delta{}t}\right\rvert,">
</p>

即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left\lvert{}1+\alpha\Delta{}t\right\rvert\cdot\left\lvert{}d_{t-\Delta{}t}\right\rvert\leq\left\lvert{}d_{t-\Delta{}t}\right\rvert.">
</p>

如果 ![](https://latex.codecogs.com/svg.latex?d_0=0) ，则数值的稳定性是平凡的（自己与自己的差值永远等于 ![](https://latex.codecogs.com/svg.latex?0) ），无需考虑。<br>
如果 ![](https://latex.codecogs.com/svg.latex?d_0\neq{}0) ，同时 ![](https://latex.codecogs.com/svg.latex?\Delta{}t=-1/\alpha) ，则

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\left\lvert{}1+\alpha\Delta{}t\right\rvert\cdot\left\lvert{}d_{t-\Delta{}t}\right\rvert&=\left\lvert{}1+\alpha\cdot\left(-\frac{1}{\alpha}\right)\right\rvert\cdot\left\lvert{}d_{t-\Delta{}t}\right\rvert\\&=0\\&\leq\left\lvert{}d_{t-\Delta{}t}\right\rvert,\end{align*}">
</p>

数值的稳定性同样是平凡的。<br>
如果 ![](https://latex.codecogs.com/svg.latex?d_0\neq{}0) ，同时 ![](https://latex.codecogs.com/svg.latex?\Delta{}t\neq-1/\alpha) ，则

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}d_{t-\Delta{}t}&=\left(1+\alpha\Delta{}t\right)\cdot{}d_{t-2\Delta{}t}\\&=\left(1+\alpha\Delta{}t\right)^2\cdot{}d_{t-3\Delta{}t}\\&\cdots\\&=\left(1+\alpha\Delta{}t\right)^{n-1}\cdot{}d_{t-n\Delta{}t}\\&\cdots\\&=\left(1+\alpha\Delta{}t\right)^{\left(t/\Delta{}t\right)-1}\cdot{}d_{t-\left(t/\Delta{}t\right)\cdot\Delta{}t}\\&=\left(1+\alpha\Delta{}t\right)^{\frac{t}{\Delta{}t}-1}\cdot{}d_{0}\\&\neq{}0,\end{align*}">
</p>

于是，不等式两边可以同时除以 ![](https://latex.codecogs.com/svg.latex?\left\lvert{}d_{t-\Delta{}t}\right\rvert) ，从而有

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left\lvert{}1+\alpha\Delta{}t\right\rvert\leq{}1,">
</p>

即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?-1\leq{}1+\alpha\Delta{}t\leq{}1,">
</p>

分成两个独立的不等式，

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?-1\leq{}1+\alpha\Delta{}t,\quad{}1+\alpha\Delta{}t\leq{}1.">
</p>

根据后一个不等式，可知

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\alpha\Delta{}t\leq{}0,">
</p>

由于 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 作为单个时间步长，是恒正的（非正的步长无意义），因此，![](https://latex.codecogs.com/svg.latex?\alpha) 必须满足 ![](https://latex.codecogs.com/svg.latex?\alpha\leq{}0) ，不能是正数，否则数值稳定性不复存在。换句话说，前向欧拉方法仅适用于那些 ![](https://latex.codecogs.com/svg.latex?\alpha) 是非正数（0或负数）的一阶线性常微分方程 ![](https://latex.codecogs.com/svg.latex?y^\prime=\alpha{}y+\beta) ，否则无法保证数值稳定性。<br>
在此基础之上，根据前一个不等式，可知

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?-2\leq\alpha\Delta{}t,">
</p>

如果 ![](https://latex.codecogs.com/svg.latex?\alpha=0) ，那么该不等式即为 ![](https://latex.codecogs.com/svg.latex?-2\leq{}0) ，是恒成立的。<br>
如果 ![](https://latex.codecogs.com/svg.latex?\alpha<0) ，则有

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{-2}{\alpha}\geq\Delta{}t,">
</p>

即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Delta{}t\leq\frac{2}{-\alpha},">
</p>

可记为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Delta{}t\leq\frac{2}{\left\lvert\alpha\right\rvert}\left(\alpha<0\right).">
</p>

总之，对于一阶线性常微分方程 ![](https://latex.codecogs.com/svg.latex?y^\prime=\alpha{}y+\beta)（ ![](https://latex.codecogs.com/svg.latex?\alpha\leq{}0) ），它只有满足以下任意一个条件，才能使用前向欧拉方法进行函数值估计：

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\alpha=0\;\,\text{or}\;\,0<\Delta{}t\leq\frac{2}{\left\lvert\alpha\right\rvert}.">
</p>

现在，回到一阶线性常微分方程

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{1}{R_mC_m}\cdot{}u_i^{(l)}(t)+\frac{du_i^{(l)}(t)}{dt}=\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t),">
</p>

将其整理为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{du_i^{(l)}(t)}{dt}=-\frac{1}{R_mC_m}\cdot{}u_i^{(l)}(t)+\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t),">
</p>

观察到

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\alpha&=-\frac{1}{R_mC_m}<0\\{}\beta&=\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t)\end{align*}">
</p>

设定时间步 ![](https://latex.codecogs.com/svg.latex?\Delta{}t)（满足 ![](https://latex.codecogs.com/svg.latex?0<\Delta{}t\leq2/\left\lvert\alpha\right\rvert) ，即 ![](https://latex.codecogs.com/svg.latex?0<\Delta{}t\leq2R_mC_m) ），应用前向欧拉方法进行估计，可得

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}u_i^{(l)}(t)&=u_i^{(l)}\left(t-\Delta{}t\right)+\Delta{}t\cdot\left[-\frac{1}{R_mC_m}\cdot{}u_i^{(l)}\left(t-\Delta{}t\right)+\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t)\right]\\&=u_i^{(l)}\left(t-\Delta{}t\right)-\frac{\Delta{}t}{R_mC_m}\cdot{}u_i^{(l)}\left(t-\Delta{}t\right)+\sum_j\frac{I_{ij}\Delta{}t}{C_m}\cdot{}s_j^{\left(l-1\right)}(t)\\&=\left(1-\frac{\Delta{}t}{R_mC_m}\right)\cdot{}u_i^{(l)}\left(t-\Delta{}t\right)+\sum_j\frac{I_{ij}\Delta{}t}{C_m}\cdot{}s_j^{\left(l-1\right)}(t),\end{align*}">
</p>

令

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\gamma=1-\frac{\Delta{}t}{R_mC_m},\quad{}w_{ij}=\frac{I_{ij}\Delta{}t}{C_m},">
</p>

同时，代入 ![](https://latex.codecogs.com/svg.latex?t=\uptau\Delta{}t) ，可得

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}u_i^{(l)}\left(\uptau\Delta{}t\right)&=\gamma\cdot{}u_i^{(l)}\left(\uptau\Delta{}t-\Delta{}t\right)+\sum_jw_{ij}\cdot{}s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right)\\&=\gamma\cdot{}u_i^{(l)}\left(\left(\uptau-1\right)\Delta{}t\right)+\sum_jw_{ij}\cdot{}s_j^{\left(l-1\right)}\left(\uptau\Delta{}t\right),\end{align*}">
</p>

`(时刻)` 可改记为 `[时间步]` ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?u_i^{(l)}\left[\uptau\right]=\gamma\cdot{}u_i^{(l)}\left[\uptau-1\right]+\sum_jw_{ij}\cdot{}s_j^{\left(l-1\right)}\left[\uptau\right].">
</p>

它就是 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)) 在离散时间步情形下的前向欧拉方法近似解（迭代式）。<br>
需要提醒一下三点：<br>
①（尤其重要，容易混淆，务必区分）<br>
对于一般的一阶线性常微分方程 ![](https://latex.codecogs.com/svg.latex?y^\prime=f(y))（其中 ![](https://latex.codecogs.com/svg.latex?f(y)=\alpha{}y+\beta) ，![](https://latex.codecogs.com/svg.latex?\beta) 与 ![](https://latex.codecogs.com/svg.latex?y) 无关，视为相对于 ![](https://latex.codecogs.com/svg.latex?y) 而言的常数），当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 的前向欧拉近似解可表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}y(t)&=y\left(t-\Delta{}t\right)+\Delta{}t\cdot{}f\left(y\left(t-\Delta{}t\right)\right)\\&=y\left(t-\Delta{}t\right)+\Delta{}t\cdot\left[\alpha{}y\left(t-\Delta{}t\right)+\beta\right]\\&=\left(1+\alpha\Delta{}t\right)\cdot{}y\left(t-\Delta{}t\right)+\beta\Delta{}t.\end{align*}">
</p>

在

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{du_i^{(l)}(t)}{dt}=-\frac{1}{R_mC_m}\cdot{}u_i^{(l)}(t)+\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t),">
</p>

实例当中，

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\beta=\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t)">
</p>

的确是关于 ![](https://latex.codecogs.com/svg.latex?t) 的函数，但该函数相对于

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?y=u_i^{(l)}(t)">
</p>

而言是无关的，应视为相对于 ![](https://latex.codecogs.com/svg.latex?y) 而言的常数。因此，该实例的当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 前向欧拉近似解应该表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)=u_i^{(l)}\left(t-\Delta{}t\right)+\Delta{}t\cdot\left[-\frac{1}{R_mC_m}\cdot{}u_i^{(l)}\left(t-\Delta{}t\right)+\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t)\right],">
</p>

绝对不应该表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)=u_i^{(l)}\left(t-\Delta{}t\right)+\Delta{}t\cdot\left[-\frac{1}{R_mC_m}\cdot{}u_i^{(l)}\left(t-\Delta{}t\right)+\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t-\Delta{}t)\right],">
</p>

这是完全错误的，没有任何依据可言。作为（相对于 ![](https://latex.codecogs.com/svg.latex?y=u_i^{(l)}(t)) 而言的）常数的

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\beta=\sum_j\frac{I_{ij}}{C_m}\cdot{}s_j^{\left(l-1\right)}(t)">
</p>

在代入前后必须保持不变。<br>
②<br>
对于

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\gamma=1-\frac{\Delta{}t}{R_mC_m},">
</p>

一般取 ![](https://latex.codecogs.com/svg.latex?0<\Delta{}t<R_mC_m) ，从而使 ![](https://latex.codecogs.com/svg.latex?0<\gamma<1) 。以下阐述理由。<br>
生理上，在没有任何外部输入电流的情况下，由于离子通道蛋白的存在以及某种生理机制所发挥的作用，膜内电荷会不断“泄漏”，使得膜内电位自发地不断下降，从而构成“膜内电位随时间推移而衰减（decay）”的现象。<br>
对于

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)=\left(1-\frac{\Delta{}t}{R_mC_m}\right)\cdot{}u_i^{(l)}\left(t-\Delta{}t\right)+\sum_j\frac{I_{ij}\Delta{}t}{C_m}\cdot{}s_j^{\left(l-1\right)}(t),">
</p>

为了实现对这一生理过程的模拟，就需要使

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?0<1-\frac{\Delta{}t}{R_mC_m}<1,">
</p>

即 ![](https://latex.codecogs.com/svg.latex?0<\Delta{}t<R_mC_m) ，从而当

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I_{\text{in}}(t)=\sum_jI_{ij}\cdot{}s_j^{\left(l-1\right)}(t)=0">
</p>

时，依据

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)=\left(1-\frac{\Delta{}t}{R_mC_m}\right)\cdot{}u_i^{(l)}\left(t-\Delta{}t\right)">
</p>

进行迭代的 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}(t)) 能够严格小于前一时刻的 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}\left(t-\Delta{}t\right)) 。<br>
![](https://latex.codecogs.com/svg.latex?\gamma\in\left(0,1\right)) 可称为“膜衰减因子”（membrane decay factor）。<br>
③<br>
原论文中 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}\left[\uptau-1\right]) 的系数符号为 ![](https://latex.codecogs.com/svg.latex?\beta) 。为避免与本篇解读笔记的前文 ![](https://latex.codecogs.com/svg.latex?f(y)=\alpha{}y+\beta) 所使用的 ![](https://latex.codecogs.com/svg.latex?\beta) 产生符号冲突，笔者在本篇解读笔记中将 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}\left[\uptau-1\right]) 的系数符号设置为 ![](https://latex.codecogs.com/svg.latex?\gamma) ，与原论文中 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}\left[\uptau-1\right]) 的系数符号 ![](https://latex.codecogs.com/svg.latex?\beta) 没有本质区别。<br>
生理上，在一般情况下，膜内电位达到一定阈值后，会通过某种生理机制进行“重置”（reset），同时向其他神经元发放脉冲。在不考虑诸多实际生理因素的理想化计算模型中，“膜内电位达到一定阈值”可视为“向其他神经元发放脉冲”的充分必要条件。<br>
为了实现对这种一般现象的建模，引入阈值电位 ![](https://latex.codecogs.com/svg.latex?\theta) 。同时，前文所述的 ![](https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]) 实际上就定义为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]=H\left(u_i^{(l)}[\uptau]-\theta\right)=\begin{cases}1,\quad{}u_i^{(l)}[\uptau]\geq\theta\\0,\quad{}u_i^{(l)}[\uptau]<\theta\end{cases}">
</p>

如果 ![](https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]=1) ，说明该神经元在时间步 ![](https://latex.codecogs.com/svg.latex?\uptau) 下的膜内电位达到或者超过了阈值 ![](https://latex.codecogs.com/svg.latex?\theta) ，那么在“软重置”（soft reset）工程实现中，这部分阈值电位需要被择机扣除。<br>
如果 ![](https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]=0) ，说明该神经元在时间步 ![](https://latex.codecogs.com/svg.latex?\uptau) 下的膜内电位没有达到阈值 ![](https://latex.codecogs.com/svg.latex?\theta) ，无需进行扣除操作。<br>
可能需要被择机扣除的量可以表示为 ![](https://latex.codecogs.com/svg.latex?-s_i^{(l)}[\uptau]\cdot\theta) 。在 ![](https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]=0) 时它等于 ![](https://latex.codecogs.com/svg.latex?0)（表示无需扣除），在 ![](https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]=1) 时它等于 ![](https://latex.codecogs.com/svg.latex?-\theta) 。<br>
实际上，前文所提到的 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau]) 在严格意义上指的是————在时间步 ![](https://latex.codecogs.com/svg.latex?\uptau) 下，对“是否发放脉冲（是否达到阈值）”进行判定**之前**的膜内电位。<br>
这就必然限定了一种明确的数值先后依赖关系：![](https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]) 值的确定依赖于 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau]) 值的确定。只有先得出 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau]) 的值，才能紧随其后判定 ![](https://latex.codecogs.com/svg.latex?s_i^{(l)}[\uptau]) 的值。<br>
在要求对连续的自然时间进行离散化建模的工程仿真实现中，这种数值先后依赖关系造成了一个必然的后果：当前时间步 ![](https://latex.codecogs.com/svg.latex?\uptau) 下判定的、可能需要被扣除的量 ![](https://latex.codecogs.com/svg.latex?-s_i^{(l)}[\uptau]\cdot\theta) ，**不得不**延迟到后一个时间步 ![](https://latex.codecogs.com/svg.latex?\uptau+1) 才能进行扣除。<br>
具体而言，以相邻时间步 ![](https://latex.codecogs.com/svg.latex?\uptau-1) 和 ![](https://latex.codecogs.com/svg.latex?\uptau) 为例，对于根据 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau-1]) 判定的 ![](https://latex.codecogs.com/svg.latex?-s_i^{(l)}[\uptau-1]\cdot\theta) ，它作为扣除项，只能放在 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau]) 迭代式的右侧，而不能放在 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau-1]) 迭代式的右侧。![](https://latex.codecogs.com/svg.latex?-s_i^{(l)}[\uptau-1]\cdot\theta) 是对 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau-1]) 值进行事后判定的结果，它本身当然是不能作为一个贡献项参与 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau-1]) 值的构建的，因为这违背了时序逻辑，会在实际的计算中引发混乱。<br>
总之，为了对膜内电位重置现象进行建模，需要在 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau]) 迭代式的右侧额外加入一个扣除项 ![](https://latex.codecogs.com/svg.latex?-s_i^{(l)}[\uptau-1]\cdot\theta) 。<br>
完整的 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau]) 迭代式即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?u_i^{(l)}[\uptau]=\gamma\cdot{}u_i^{(l)}\left[\uptau-1\right]+\sum_jw_{ij}\cdot{}s_j^{\left(l-1\right)}[\uptau]-s_i^{(l)}\left[\uptau-1\right]\cdot\theta,">
</p>

其中

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\gamma=1-\frac{\Delta{}t}{R_mC_m}\in\left(0,1\right)">
</p>

是膜衰减因子，

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?w_{ij}=\frac{I_{ij}\Delta{}t}{C_m}">
</p>

表示突触权重（synaptic weight），![](https://latex.codecogs.com/svg.latex?\theta) 表示阈值电位。<br>
原论文作者团队指出：<br>
“我们使用迭代性的漏电-积分-发放（LIF）形式化框架来建模脊髓中间神经元（spinal interneurons）。关键的是，该网络以状态化的膜动力学（stateful membrane dynamics）进行实例化，这意味着膜电位 ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}) 在经过一个接着一个时间步期间（across successive time steps）被严格保留，而非在每次前向传播时重新初始化。这赋予了脊髓基底隐式的时序工作记忆（implicit temporal working memory），使其能够编码历史依赖性的特征，而无需显式的循环门控单元（例如LSTM）。”<br>
“ ![](https://latex.codecogs.com/svg.latex?u_i^{(l)}\left[\uptau-1\right]) 项代表从前一片刻携带过来的残余电位（residual potential），这建立了一个连续的时序上下文（continuous temporal context）。”
