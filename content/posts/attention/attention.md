---
title: "Calculate Attention"
date: 2024-12-22T00:31:04+08:00
summary: "實作參考[Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html): Implementation and Concept of Attention\n"
tags: ["Attention"]
---


實作參考[Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


## Scaled Dot-Product Attention

<p>$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$</p>
<img src="https://hackmd.io/_uploads/rJuXRe24ye.png" style="margin:auto;display:block;">

<h3> 關於 \(\sqrt{d_k}\) </h3>

1. <p>為何需要縮放因子 \(\sqrt{d_k}\) ? 這邊用例子簡單說明</p>

    - 如果有一個值 `x[3]` 大其他值很多，經過softmax會將 `x` 範圍變到從0到1之間的 `y`，除了 `y[3]` 的其他值就會都趨近0(變得很不重要)，造成
        ```python
        x = torch.FloatTensor([1, 1, 1, 5])
        y = torch.softmax(x, dim=-1)
        # y: [0.0174, 0.0174, 0.0174, 0.9479]
        ```
    - 畫成圖可能就會長這樣
        <img src="https://hackmd.io/_uploads/H1x1SlnN1g.png" width=300 style="display:block;">

2. 所以我們需要一個縮放因子縮小這個差異，這邊可以回憶一下[標準化](https://zh.wikipedia.org/zh-tw/%E6%A0%87%E5%87%86%E5%8C%96_(%E7%BB%9F%E8%AE%A1%E5%AD%A6))的原理，就是把資料分布改得較符合常態分佈，並縮小離群值對模型的影響，平均值為0，且標準差為1
    $$Z = \frac{X-\mu}{\sigma} \sim N(0,1)$$
    > :bulb: [標準化VS.正規化](https://ithelp.ithome.com.tw/articles/10293893)

3. <p>那這個重責大任為啥落到 \(\sqrt{d_k}\) 身上呢?</p>
    
    - <p>先來看paper怎麼解釋: 假設 \(q\) 和 \(k\) 是常態分佈( \(\mu=0\)、\(\sigma=1\) )，那麼 \(q \cdot k\) 的 \(\mu=0\) 且 \(\sigma=d_k\) → 在說啥</p>

        ><p>To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, \(q \cdot k = \textstyle\sum_{i=0}^{d_k} q_i k_i\) , has mean 0 and variance \(d_k\).</p>
        
    - <p>先解決第一句:  <b>假設 \(q\) 和 \(k\) 是常態分佈( \(\mu=0\)、\(\sigma=1\) )</b></p>
        期望值 <span>\(E[q_i \cdot k_i] = 0\)</span>
        
        標準差 
        <p>\(Var(q_i k_i) = 1\)</p>
        <p>\(\implies E[q_i^2 k_i^2] - E^2[q_i k_i] = 1\)</p>
        <p>\(\implies E[q_i^2] E[k_i^2] - E^2[q_i] E^2[k_i] = 1\)</p>
        <p>\(\implies ( E[q_i^2] - E^2[q_i]) ( E[k_i^2] - E^2[k_i]) - E^2[q_i] E^2[k_i] = 1\)</p>
        <p>\(\implies Var(q_i) \ Var(k_i) - 0 = 1\)</p>
        <p>\(\implies Var(q_i) \ Var(k_i) = 1\)</p>
        
        
    - <p>再解決第一句: \(q \cdot k\) 的 \(\mu=0\) 且 \(\sigma=d_k\)</p>
            
        <p>$$
            \begin{aligned}
            E[q \cdot k] &= \textstyle\sum_{i=0}^{d_k} E[q_i k_i] \\
            &= d_k \times0 \\ 
            &=0
            \end{aligned}
        $$</p>
        
        <p>$$\begin{aligned}
        Var(q \cdot k)&=\textstyle\sum_{i=0}^{d_k} Var(q_i k_i)\\
        &= d_k \times1 \\
        &= d_k \\
        \end{aligned}$$</p>

    - :tada: 太棒了!破案了!
        最後套上標準化的公式
        <p>$$\begin{aligned}
        Z&=\cfrac{QK^T - E[QK^T]}{\sqrt{Var(QK^T)}} \\
        &=\cfrac{QK^T - 0}{\sqrt{d_k}}\\
        &= \cfrac{QK^T}{\sqrt{d_k}}
        \end{aligned}$$</p>
    :bulb: 最後 <span>\(\cfrac{QK^T}{\sqrt{d_k}}\)</span> 就可以使attention map經過 Softmax 梯度就不會被削弱了!
- 更詳細數學可以看
    - 比較不同角度解釋: [为什么在进行softmax之前需要对attention进行scaled（为什么除以 $d_k$的平方根）](https://blog.csdn.net/ytusdc/article/details/121622205)
    - 有推導數學(推推): [分析與拓展：Transformer中的MultiHeadAttention為什麼要用scaled？](https://allenwind.github.io/blog/16228/)

### 如何知道 $\sqrt{d_k}$ 多大
- 三個要關注的attention來源: query、value、key的維度可以當成都是一樣的，參考以下[這張圖](https://miro.medium.com/v2/resize:fit:828/format:webp/1*m58HPvaWXAt1bYNEHSwycA.png)
    ![image](https://hackmd.io/_uploads/r1I7hAoEkg.png)
- 關於數學式視覺化後可以參考[這張圖](https://miro.medium.com/v2/resize:fit:828/format:webp/1*amnlT6Hjm5nV6NjFI0tRyg.png)
    ![image](https://hackmd.io/_uploads/S1E5nAj4ye.png)
- 加上batch的維度，query、key、value變成tensor後的維度會長這樣
    ```
    query: (batch_size, num_heads, seq_length, d_q)
    key: (batch_size, num_heads, seq_length, d_k)
    query: (batch_size, num_heads, seq_length, d_v)
    ```
    > :bulb: `num_heads` 因為有多個Multi-Head Attention
    > <img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_38_0.png" width=300>

- <p>所以獲取 \(d_k\) 的方式就是取得 <code>query</code> 的最後一個維度(反正 <code>d_q</code> = <code>d_k</code> = <code>d_v</code> )</p>

    ```python
    d_k = query.size(-1)
    ```
### 計算Attention分數
- Key的Transpose(可以去複習線代就大概知道這是一個怎麼回事)就是將最後一個維度和倒數第二個維度翻轉
    ```python
    key = key.transpose(-2,-1) 
    
    -> key: (batch_size, num_heads, d_k, seq_length)
    ```
- dot product
    - 為何要用dot product來看相關性呢?參考[這張圖](https://zilliz.com/blog/similarity-metrics-for-vector-search)
       ![image](https://hackmd.io/_uploads/rkhfFWTEkl.png)
    - <p>\(\theta\) 越大 → dot product 越小 → 代表越不相關</p>
    - <p>\(\theta\) 越小 → dot product 越大 → 代表越相關</p>
- 接下來就是query和key、以及value做矩陣相乘算出attention分數，
    - 我個人覺得可以這樣理解\
        ▶ query: 搜尋詞
        
        ▶ 搜出來的的資料，每個資料會長這樣 
        ```
        {key1: value1, key2: value2, ..., keyN:valueN}
        key: 資料的標題
        value: 資料內容
        ```
            
        ▶ 注意力機制就是要看哪個資料跟我的搜尋詞最相關，於是用dot product來看他們相關性的分數
        ```
        attention = query * key
        ```        
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➜ attention 算出來就是每筆資料的標題和我搜尋的相關分數
        ![image](https://hackmd.io/_uploads/S1_K613VJx.png)\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➜ 分數越高，代表越接近我在搜尋的東西
            
        ▶ 最後在將這個分數乘上value，啥意思?\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➜ 給每筆資料打完相關性的分數後，在乘上資料內容就可以取得那些是重要資料內容了\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➜ 像這樣，把每筆資料的分數乘上資料內容，就可以知道如果分數較大，資料內容乘上分數就會比較大\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➜ 也代表著此內容較重要的意義，相反則同樣意義\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➜ 量化後就是一堆數字相乘，以下大概就是分數和資料內容矩陣相乘的樣子\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://hackmd.io/_uploads/H1He912Vkg.png" width=600>\
      
    - 所以程式碼實現就是以下
        ```python
        scores =  torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        ```
    >:bulb: 為何是query與key相乘的細節可以參考[李宏毅的教材](https://www.youtube.com/watch?v=hYdO9CscNes)


## Real World Example
到 [GitHub](https://github.com/Hlunlun/Transformer/blob/master/example/scaled_dot_product.ipynb) 看實際數據模擬

## Implementation
完整的 Scaled Dot-Product Attention 實作
```python
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```







## Reference
- [Tutorial on Scaled Dot-Product Attention with PyTorch Implementation from Scratch](https://medium.com/@vmirly/tutorial-on-scaled-dot-product-attention-with-pytorch-implementation-from-scratch-66ed898bf817)
- [Why use a "square root" in the scaled dot product](https://ai.stackexchange.com/questions/41861/why-use-a-square-root-in-the-scaled-dot-product)

