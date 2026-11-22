# 方法一 F1=79.7998

## 一、对英文数据的训练

### 1、建立分类头（本质是个线性层）

英文训练数据本来就有 sense_id，读取已有标签，建立分类头，一个词语有几个意思就建立一个输出几个向量的分类头，同时每个词都有自己的分类头，但它们共享同一个 XLM-R 编码器。

### 2．把句子转换为目标词向量

输入第一句话：

She deposited money at the bank.

XLM-R 编码整句话，为每个子词输出一个向量，程序根据 start/end 找到 bank 对应的向量。

原 joint 的训练池化方式是：

1. 对最后四层的输出取平均。

2. 对目标词对应的子词取平均。

3. 将结果归一化。

得到：$z_{bank} \in \mathbb{R}^{1024}$

虽然只取 bank 位置的输出，但这个向量已经融合了整句上下文。 因此，“存钱句中的 bank”和“河边句中的 bank”可以得到不同向量。

### 3、分类头预测当前词义

将$z_{bank}$输入 bank 专属的分类头：![分类头矩阵公式](data:image/png;base64,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)

假设这里的bank只有两个词义。

输出两个分数，经 softmax 转换为概率，这里会产生分类损失，采用交叉熵损失函数，取的是真实类别的预测概率，因为该数据有标签。

```python
label_map = {
    "finance": 0,    # 银行
    "riverside": 1,  # 河岸
}
```

$$s = \begin{bmatrix}
s_{0} \\
s_{1}
\end{bmatrix} = \begin{bmatrix}
银行的分数 \\
河岸的分数
\end{bmatrix}$$

| **预测情况**     | **银行的概率** | **河岸的概率** | **交叉熵损失**               |
|------------------|----------------|----------------|------------------------------|
| 不太相信正确答案 | 0.40           | 0.60           | $$- \ln 0.40 \approx 0.916$$ |
| 很相信正确答案   | 0.90           | 0.10           | $$- \ln 0.90 \approx 0.105$$ |

然后训练通过反向传播调整参数，推动“银行”的概率提高。

### 4、三元组损失直接约束向量之间的关系

（1）分类预测正确，还不等于向量一定适合余弦距离聚类。因此，原 joint 同时使用度量损失。）

（2）做瑞典语聚类时，我们会去掉英文分类头，直接比较目标词向量的余弦距离。（那为啥还要分类头呢？直接做余弦距离聚类）

答：因为英文训练词的分类头，只认识那个英文词的那套标签。例如 bank 的分类头只输出 finance/riverside，不能拿它判断瑞典语 skär 的词义，但所有词都共享 XLM-R。训练不同词时，损失会共同调整这个编码器，让它有机会学到更通用的“根据上下文区分目标词用法”的能力。

| **角色** | **例句**           | **与锚点的关系** |
|----------|--------------------|------------------|
| 锚点 A   | 在 bank 存钱       | —                |
| 正例 P   | bank 批准贷款      | 同义             |
| 负例 N   | 坐在河流的 bank 上 | 异义             |

A该向量（XLM-R 编码整句话，为每个子词输出一个向量，程序根据 start/end 找到 bank 对应的向量）与P、N分别计算余弦相似度。

$$s_{AP} = \cos(z_{A},z_{P}),s_{AN} = \cos(z_{A},z_{N})$$

要求同义对比异义对更相似，并留出 margin=0.2（人为设定0.1-0.3还没有试过） 的间隔：

$$L_{A} = \max_{}(0,\mspace{6mu} 0.2 - s_{AP} + s_{AN})$$

等于0即，这组关系已经满足间隔要求。

“双向”还要检查正例 P 与负例 N 的关系

$$L_{P} = max(0,\mspace{6mu} 0.2 - s_{AP} + s_{PN})$$

$$L_{metric} = 1/2\left\lbrack 1/3\max(0,\mspace{6mu} 0.2 - s_{AP} + s_{AN_{k}}) + 1/3\sum_{k = 1}^{3}\max(0,\mspace{6mu} 0.2 - s_{AP} + s_{PN_{k}}) \right\rbrack$$

原代码每组实际采样三个负例，分别计算两种约束并取平均：

### 5．合并损失，反向传播

$L = L_{metric} + 0.2L_{CE}$（0.1-0.3都试过，结果相同）

## 二、对瑞典语的聚类

### 6．瑞典语 WSI：提取向量，再聚类

使用英文词义标注数据微调好的模型，来提取瑞典语目标词向量。

训练结束后，对瑞典语每条用法：

1. 输入原句。

2. 根据目标位置提取向量。

3. 同一个词的所有用法计算余弦距离。

4. 使用 complete 层次聚类，得到簇编号。

假设只有四个用法 A、B、C、D，初始就是四个簇：

$$\{ A\},\quad\{ B\},\quad\{ C\},\quad\{ D\}$$

complete 用“最远的一对”定义簇间距离

现在判断 {A,B} 和{C}能不能合并。假设：

$$d(A,C) = 0.30,\quad\quad d(B,C) = 0.70$$

用阈值 0.650 决定最终保留哪些簇。

方法二 F1=75.6342

1、mBERT 分支对目标词对应的子词向量进行平均池化，获得目标词表示

2、BGE-M3 分支以目标词表示作为查询，计算其与句中各词元的余弦相似度并通过 softmax 得到注意力权重，对包含目标词在内的上下文向量进行加权汇总。

例如句子：

银行批准了贷款。

以目标词“银行”为关注中心，假设算出的注意力权重是：

| **词元** | **权重** |
|----------|----------|
| 银行     | 0.40     |
| 批准     | 0.25     |
| 了       | 0.05     |
| 贷款     | 0.30     |

那么加权汇总就是：$z = 0.40h_{银行} + 0.25h_{批准} + 0.05h_{了} + 0.30h_{贷款}$

（实际权重由目标词向量与各词元向量的相似度计算得到）

3、两个分支分别计算同一单词不同用法之间的余弦距离，并对距离矩阵进行加权融合，最后使用 complete 层次聚类，将语义相近的用法划分到同一簇，实现词义归纳。

| **方法**           | **宏平均 F1** |
|--------------------|---------------|
| mBERT＋BGE-M3 融合 | 0.756342      |
| 原 joint           | **0.797978**  |
| 差值               | −0.041637     |
