---
title: "Multi-Head Attention"
date: 2024-12-21T23:49:08+08:00
summary: "實作參考[Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
: Implementation and concept of Multi-Head Attention"
tags: ['Attention']
---


實作參考[Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


<img src="https://hackmd.io/_uploads/B1IzQb3Eye.png" style="display:block;margin:auto;" width=300>


## 數學
<p>
$$
MultiHead(Q,K,V)=Concat({head}_1, {head}_2, ..., {head}_h)
$$ 
</p>

<p>
$${head}_i = Attention(Q{W_i}^Q, K{W_i}^K, V{W_i}^V)$$
</p>


- <p>可學習的參數就是其中Query的權重 \(W_i^Q\) 、Key的權重 \(W_i^K\) 、Value的權重 \(W_i^V\) </p>
- 忘記Attention是怎麼算的可以看[這裡](https://hackmd.io/@clh/transformer-attention-0)
- 結合算Attention的過程，整個Multi-Head Attention大概是以下這樣算


## 從程式碼看
### 全連接層 `nn.Linear(d_model, d_model)`
```python
self.linears = clones(nn.Linear(d_model, d_model), 4)
```
1. 輸入輸出維度: 整個模型的的維度
    ```python
    output: (batch_size, seq_length, d_model)
    ```
2. 總共需要四個全連接層
    - K、Q、V各一個
        <img src="https://hackmd.io/_uploads/rken74T4ye.png" style="display:block;">
    - 最後輸出再一個
        <img src="https://hackmd.io/_uploads/H14J4ET4yx.png" style="display:block;">
3. 經過 全連接層 的V, K, Q 就會長這樣
    <img src="https://hackmd.io/_uploads/SJJwrNpVyg.png" style="display:block;">
    - <p> \(W_i\) 一開始就是隨機初始化的的權重，在訓練過程中模型會自己用loss function來back propagate來學習並更新這些參數，就可以學習到最好的權重</p>
    - 關於全連接層詳細參考[李弘毅講義](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/regression%20(v16).pdf)

<h3> Q, K, V 透過 Linear 映射到 \(Q{W_i}^Q\), \(K{W_i}^K\), \(V{W_i}^V\) </h3>

```python
# Do all the linear projections in batch from d_model => h x d_k
query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key,value))]
```

1. 先講 `view()`
    - <p><code>view(nbatches, -1, self.h, self.d_k)</code>: 對 輸出的張量 \(W_i x_i\) 進行reshape</p>
        <!-- ```
        view(nbatches, -1, self.h, self.d_k) 是对输出张量进行重塑（reshape）的操作。
        参数解释：
        nbatches: 保持批次的大小不变。
        -1: 表示自动推断这一维度的大小，以保持总元素数量不变。在这里，它将根据输入的总元素数量自动计算出序列长度（seq_length）。
        self.h: 表示头的数量。
        self.d_k: 每个头的维度（即每个头处理的特征维度）。
        功能：这个操作将张量重塑为四维形式，便于后续多头注意力计算。重塑后的形状为 [nbatches, seq_length, h, d_k]。
        ``` -->
2. 關於各種維度，用query Q 舉例，shape = (head, seq_length, d_k) = (2, 3, 2)
    <p>
    $$q_0=\begin{bmatrix}
       1.45 & -0.79 \\
       0.62 & -0.64 \\
       0.97 & -0.21 \\ 
    \end{bmatrix}$$    
    $$q_1=\begin{bmatrix}
       0.40 & 0.53 \\
       0.40 & 0.29 \\
       0.48 & 0.80 \\ 
    \end{bmatrix}$$
    </p>
    <p>
        <ul>
            <li>\(d_k\) = 2，也就是 \(q_i\) 的行數</li>
            <li> <code>seq_lenghth</code> = 3，<span>\(q_i\)</spn> 的列數</li>
            <li><code>head</code> = 2，也就是 \(Q\) 的長度 2，因為有兩個 子陣列</li>
        </ul>
    </p>  
3. 整個過程的維度可以參考[這張圖](https://www.kaggle.com/code/aisuko/coding-cross-attention)
    <img src="https://files.mastodon.social/media_attachments/files/111/819/100/204/990/207/original/09d504e16eb77ccc.png" width=600 style="display:block">


## Real World Example
可以到 [GitHub](https://github.com/Hlunlun/Transformer/blob/master/example/multi_head_attention.ipynb) 看實際數據模擬

## 完整實作
```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super(MultiHeadedAttention, self).__init__()
        assert d_model%h == 0
        # Assume d_v always d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implement multi-head attention"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key,value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear[-1](x)
```


## Reference
- [Self -Attention、Multi-Head Attention、Cross-Attention](https://blog.csdn.net/philosophyatmath/article/details/128013258)
- [详解Transformer中Self-Attention以及Multi-Head Attention](https://blog.csdn.net/qq_37541097/article/details/117691873)


