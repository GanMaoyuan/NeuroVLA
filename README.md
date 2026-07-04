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
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}K&=W_K\cdot\text{Concat}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right)\in\mathbb{R}^{MS\times{}D}\\{}V&=W_V\cdot\text{Concat}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right)\in\mathbb{R}^{MS\times{}D}\end{align*}">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?W_K,W_V\in\mathbb{R}^{MS\times{}MS}) 为投影矩阵。

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
为了在本篇解读笔记中严格区分时刻 ![](https://latex.codecogs.com/svg.latex?t) 与时间步 ![](https://latex.codecogs.com/svg.latex?\uptau)（相邻时刻之间更细粒度的时间尺度）这两种不同的概念，笔者在以下对GRU的描述过程中采用 ![](https://latex.codecogs.com/svg.latex?t^\prime) 来描述时刻，而非 ![](https://latex.codecogs.com/svg.latex?\uptau) 。

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

所原本包含的信息。同时，与之相比，![](https://latex.codecogs.com/svg.latex?\mathbf{h}_t\in\mathbb{R}^{D_h}) 是紧凑的（compact）——有关物理状态本体感觉的信息原本由一个矩阵包含，现在则由一个元素数量更少的向量包含。这实现了信息的压缩。<br>
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

计算一组仿射变换参数——缩放参数 ![](https://latex.codecogs.com/svg.latex?\gamma_t) 和偏移参数 ![](https://latex.codecogs.com/svg.latex?\beta_t) ，即

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

# 小脑模块——迭代循环优化

通过某种机制，可以获得未来时刻 ![](https://latex.codecogs.com/svg.latex?t+1)（未来的第一个时刻）下的本体感觉预期状态（anticipated state）![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}\in\mathbb{R}^{D_s}) 。<br>
需要指出的是，原文中仅提到 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}) 本身，并未给出其产生的具体机制。笔者选择将其形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}=\text{Anticipate}\left(\text{unknown}\;\,\text{inputs}\right).">
</p>

定义 ![](https://latex.codecogs.com/svg.latex?\text{Refine}\left(\cdot,\cdot\right)) 模块，它封装了GRU子模块和Gated FiLM子模块的计算流程。它对输入进行循环迭代，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{\left(i+1\right)}\leftarrow\text{Refine}\left(\mathbf{z}_{\text{mod}}^{(i)},\mathbf{s}_{t+1}^{(i)}\right),">
</p>

本体感觉预期状态也将同步地循环迭代，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}^{\left(i+1\right)}\leftarrow\text{Update}\left(\mathbf{s}_{t+1}^{(i)},\text{other}\;\,\text{unknown}\;\,\text{inputs}\right).">
</p>

设定循环次数为 ![](https://latex.codecogs.com/svg.latex?I) ，以上两式中的 ![](https://latex.codecogs.com/svg.latex?i) 便满足

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?i\in\{0,1,2,\cdots,I-1\}.">
</p>

以 ![](https://latex.codecogs.com/svg.latex?i=0) 为例，第一次迭代产生 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{(1)}\in\mathbb{R}^K) 的全过程

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{\left(1\right)}\leftarrow\text{Refine}\left(\mathbf{z}_{\text{mod}}^{(0)},\mathbf{s}_{t+1}^{(0)}\right)">
</p>

可具体描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}\mathbf{S}_{t-H+2:t+1}^{(0)}&=\text{Concat}\left(\mathbf{S}_{t-H+2:t},\mathbf{s}_{t+1}^{(0)}\right)\\{}&=\text{Concat}\left(\mathbf{S}_{t-H+2:t},\mathbf{s}_{t+1}\right)\in\mathbb{R}^{H\times{}D_s}\\{}\mathbf{h}_{t+1}^{(0)}&=\text{GRU}\left(\mathbf{h}_{t-H+1},\mathbf{S}_{t-H+2:t+1}^{(0)}\right)\in\mathbb{R}^{D_h}\\{}\mathbf{g}_{t+1}^{(0)}&=\sigma\left(W_g\cdot\text{Proj}\left(\mathbf{h}_{t+1}^{(0)}\right)\right)\in\mathbb{R}^{D_{\text{action}}}\\{}\gamma_{t+1}^{(0)}&=f_\gamma\left(\mathbf{h}_{t+1}^{(0)}\right)\in\mathbb{R}^K,\quad{}\beta_{t+1}^{(0)}=f_\beta\left(\mathbf{h}_{t+1}^{(0)}\right)\in\mathbb{R}^K\\{}\mathbf{z}_{\text{mod}}^{(1)}&=\left(1+\gamma_{t+1}^{(0)}\right)\odot\left(\mathbf{z}_{\text{mod}}^{(0)}\cdot\mathbf{g}_{t+1}^{(0)}\right)+\beta_{t+1}^{(0)}\\{}&=\left(1+\gamma_{t+1}^{(0)}\right)\odot\left(\mathbf{z}_{\text{mod}}\cdot\mathbf{g}_{t+1}^{(0)}\right)+\beta_{t+1}^{(0)}\in\mathbb{R}^K\end{align*}">
</p>

同时，本体感觉预期状态发生更新，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}^{(1)}\leftarrow\text{Update}\left(\mathbf{s}_{t+1}^{(0)},\text{other}\;\,\text{unknown}\;\,\text{inputs}\right).">
</p>

应注意，第一次迭代时，![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}^{(0)}) 指的就是 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}_{t+1}) ，![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}^{(0)}) 指的就是 ![](https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{mod}}) 。
