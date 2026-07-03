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
总之，如果将键/值空间投影、交叉注意力、仿射这三部分计算逻辑封装在 ![](https://latex.codecogs.com/svg.latex?\text{Q-Former}(\cdot,\cdot)) 函数内部，则Q-Former子模块整体的输出结果可以形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{\text{sem}}=\text{Q-Former}\left(\text{Concat}\left(\mathcal{H}_t\left[l_{\text{start}}:l_{\text{end}}\right]\right),Q\right)\in\mathbb{R}^{K\times{}D_{\text{action}}}.">
</p>

<br><br><br><br><br><br><br><br>

# 小脑模块——GRU子模块

机器人的传感器每进行一次采样，就会得到一个反映机器人物理状态本体感觉（proprioception）的状态向量 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}\in\mathbb{R}^{D_{\text{state}}}) ，它的 ![](https://latex.codecogs.com/svg.latex?D_{\text{state}}) 个分量表征了机器人的关节角度（joint angles）、速度（velocities）、六自由度力旋量（6-DoF wrench measurements）。<br>
在后文中，![](https://latex.codecogs.com/svg.latex?D_{\text{state}}) 简记为 ![](https://latex.codecogs.com/svg.latex?D_s) 。<br>
传感器的采样周期为 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) ，即每隔 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 秒便会进行 ![](https://latex.codecogs.com/svg.latex?1) 次采样。![](https://latex.codecogs.com/svg.latex?\Delta{}t) 往往是毫秒级的，例如 ![](https://latex.codecogs.com/svg.latex?\Delta{}t=0.02s=20ms) 。传感器的采样频率为 ![](https://latex.codecogs.com/svg.latex?f) ，即 ![](https://latex.codecogs.com/svg.latex?1) 秒钟之内总共进行 ![](https://latex.codecogs.com/svg.latex?f) 次采样（例如 ![](https://latex.codecogs.com/svg.latex?f=50\mathrm{Hz}) ）。<br>
既然每隔 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 秒就会进行 ![](https://latex.codecogs.com/svg.latex?1) 次采样，那么在 ![](https://latex.codecogs.com/svg.latex?1) 秒钟之内，采样的总次数就自然是 ![](https://latex.codecogs.com/svg.latex?1/\Delta{}t) 。因此，![](https://latex.codecogs.com/svg.latex?f) 与 ![](https://latex.codecogs.com/svg.latex?\Delta{}t) 的关系即为 ![](https://latex.codecogs.com/svg.latex?f=1/\Delta{}t) 。<br>
设定传感器的历史状态（History State）保留窗口的时间步长度为 ![](https://latex.codecogs.com/svg.latex?H) 。也可以称之为“状态时域/状态的时间跨度”（State Horizon）。<br>
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
此外，笔者将字母另改为大写形式 ![](https://latex.codecogs.com/svg.latex?\mathbf{S}) ，是为了对单个历史状态向量（用小写的 ![](https://latex.codecogs.com/svg.latex?\mathbf{s}) ）与包含了多个历史状态向量的时序矩阵（用大写的 ![](https://latex.codecogs.com/svg.latex?\mathbf{S}) ）进行区分。<br>
首先声明，仅就当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 而言，GRU（Gated Recurrent Unit，门控循环单元）在历史时刻 ![](https://latex.codecogs.com/svg.latex?t-H) 的隐藏状态 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}}) 是已知的，属于当前时刻 ![](https://latex.codecogs.com/svg.latex?t) 下可以直接使用的已知量。不过，如果当前时刻为 ![](https://latex.codecogs.com/svg.latex?t=0) ，也就是“系统刚刚出厂开机”，并没有先前计算好的已知量可供使用，则不妨将 ![](https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}\in\mathbb{R}^{D_{\text{hidden}}}) 初始化为零向量以供后续计算，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{h}_{t-H}=\mathbf{0}\in\mathbb{R}^{D_{\text{hidden}}}.">
</p>

![](https://latex.codecogs.com/svg.latex?D_{\text{hidden}}) 表示GRU的隐藏层（hidden layer）维度。在后文中，![](https://latex.codecogs.com/svg.latex?D_{\text{hidden}}) 简记为 ![](https://latex.codecogs.com/svg.latex?D_h) 。<br>
为了在本篇解读笔记中严格区分时刻 ![](https://latex.codecogs.com/svg.latex?t) 与时间步 ![](https://latex.codecogs.com/svg.latex?\uptau)（相邻时刻之间更细粒度的时间尺度）这两种不同的概念，笔者在以下对GRU的描述过程中采用 ![](https://latex.codecogs.com/svg.latex?t^\prime) 来描述时刻，而非 ![](https://latex.codecogs.com/svg.latex?\uptau) 。<br>
GRU的更新门（Update Gate）计算可形式化描述为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbf{z}_{t^\prime}=\sigma\left(W_{\mathbf{z}}\mathbf{s}_{t^\prime}+U_{\mathbf{z}}\mathbf{h}_{t^\prime-1}\right)\in\mathbb{R}^{D_h},">
</p>

其中

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?t^\prime\in\{t-H+1,\cdots,t\}">
</p>

表示当前时刻，此外

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?W_{\mathbf{z}}\in\mathbb{R}^{D_h\times{}D_s},\quad{}\mathbf{s}_{t^\prime}\in\mathbb{R}^{D_s},\quad{}U_{\mathbf{z}}\in\mathbb{R}^{D_h\times{}D_h},\quad{}\mathbf{h}_{t^\prime-1}\in\mathbb{R}^{D_h},">
</p>

![](https://latex.codecogs.com/svg.latex?\sigma(\mathbf{x})) 是Sigmoid函数，它会对输入向量 ![](https://latex.codecogs.com/svg.latex?\mathbf{x}) 的每一个分量 ![](https://latex.codecogs.com/svg.latex?x_i) 独立地计算

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\sigma\left(x_i\right)=\frac{1}{1+e^{-x_i}},">
</p>

