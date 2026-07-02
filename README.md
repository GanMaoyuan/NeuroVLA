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
