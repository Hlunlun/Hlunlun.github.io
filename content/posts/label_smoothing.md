---
title: "Label Smoothing"
date: 2025-02-24T22:08:20+08:00
summary: 一般是用在分類任務，可以降低 overfitting 的情況
tags: ["transformer"]
---




## Why?
![image](https://hackmd.io/_uploads/HJO3cLtc1g.png)

一般是用在分類任務，假如今天是用 cifar10 來訓練模型，你有10個標籤，即 `label = [0,1,2,3,4,5,6,7,8,9]`，再輸入模型後，模型會預測這張圖片在十個標籤中的機率，可能會出現以下的狀況
- `pred = [0,0,0,1,0,0,0,0,0,0]`

也就是說模型會非常肯定這張圖片一定就是第3個類別，那這樣為什麼是不好的呢?

因為我們不希望模型 overfitting，而overfitting的原因就是模型被答案的狀況太嚴重，造成他對很多圖片的判斷都非常自信，會造成他只把訓練即每張圖片對應的標籤都背起來但是對沒有看過的圖片(test data)就完全不知道怎麼辦

所以，為了增加模型的泛化能力，也就是希望模型可以更 general，而不是只能作答已經看過的問題，可以把答案做平滑的處理，如
- `pred = [0.1,0,0.02,0.75,0,0.1,0,0,0.03,0]`

機率最大的依然是第三個類別，但是其他標籤還是有可能性，讓模型不要太相信一定是第三類別，這樣就可以降低 overfitting 的情況了!


## 傳統 one-hot label 的數學意義
1. <p>首先可以看到計算 likelihood 的的公式，\(k\) 是第幾個類別，\(w_k\)是權重，\(x^T\)是倒數第二層的激活值，\(L\)我猜應該是類別總數假如總共有10個類別，那以下就是代表在10個類別中的機率</p>

$$p_k=\cfrac{e^{x^Tw_k}}{\sum_{l=1}^L{e^{x^Tw_l}}}$$

2. 計算 Cross-Entropy 的公式如以下，我們的目標就是縮小這個loss的值，但這跟我們一般看到算loss的公式不太一樣? 
    
    $$H(y,p) = \sum_{k=1}^K -y_klog(p_k)$$
    
    <p>
    $$y_k = \begin{cases} 
    1, & \text{if } k \text{ is the correct class} \\
    0, & \text{otherwise}
    \end{cases}$$
    </p>

    <p>這邊可以簡單舉一個 cifar10 的例子，所以總類別數是 \(K=10\)， 假設 \(y_2=1\)，loss就是以下這樣</p>
    
    <p>$$
        \begin{aligned}
        H(y, p) &= \sum_{k=1}^K -y_k \log(p_k) \\
        &= 0 \cdot \log(p_1) - 1 \cdot \log(p_2) - 0 \cdot \log(p_3) - \cdots \\
        &= -\log(p_2) \\
        &= \log\left(\frac{1}{p_2}\right)
        \end{aligned}
    $$</p>



    
    <p>也就是說，只要 \(p_2\) 越接近 \(1\) ，\(log(\frac{1}{p_2})\) 就越接近0，就達成我們想要降低 loss 的目標!</p>
           


## Label Smoothing
1. 平滑 label 的分布，可以把這段扣畫出平滑前後的分布，其實就是把一些可能性均分給其他label，讓模型不要太果斷
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # 參數設置
    K = 5  # 總類別數
    alpha = 0.1  # 平滑參數
    correct_class = 2  # 假設正確類別是索引 2（從 0 開始）

    # 1. 未平滑的硬標籤 (one-hot)
    hard_label = np.zeros(K)
    hard_label[correct_class] = 1.0

    # 2. 平滑後的標籤
    smoothed_label = np.zeros(K)
    smoothed_label[correct_class] = 1.0 * (1 - alpha) + alpha / K  # 正確類別
    smoothed_label[hard_label == 0] = alpha / K  # 其他類別

    # 3. 創建數據
    categories = np.arange(K)  # 類別索引 (0, 1, 2, 3, 4)
    width = 0.35  # 柱子的寬度

    # 4. 繪製柱狀圖
    plt.figure(figsize=(10, 6))
    plt.bar(categories - width/2, hard_label, width, label='Hard Label (No Smoothing)', color='blue')
    plt.bar(categories + width/2, smoothed_label, width, label=f'Smoothed Label (α = {alpha})', color='orange')

    # 5. 設置圖表屬性
    plt.xlabel('Class Index')
    plt.ylabel('Probability')
    plt.title('Distribution Before and After Label Smoothing')
    plt.xticks(categories, [f'Class {i}' for i in categories])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. 顯示圖表
    plt.savefig('label_smoothing_comparison.png')
    plt.show()
    ```
      
    ![image](https://hackmd.io/_uploads/rkFGovt91l.png)


2. <p>數學公式，將 \(y_k\) 修改成 label smoothing 的 label 如下， \(\alpha\) 數值範圍 [0,1] 可以自己設</p>

    $$y_k^{LS} = y_k \cdot (1-\alpha) + \alpha/K$$
    
    <p>現在假設總共有3個類別，如果正確的label是 \(k=1\)，設定 \(\alpha=0.1\)</p>
    
    $$y_1^{LS} = 1 \cdot (1-0.1) + 0.1/3 = 0.9 + 0.0333 = 0.9333$$
    
    $$y_2^{LS} = 0 \cdot (1-0.1) + 0.1/3 = 0 + 0.0333 = 0.0333$$ 
    
    $$y_3^{LS} = 0 \cdot (1-0.1) + 0.1/3 = 0 + 0.0333 = 0.0333$$
    
    <p>正確的 \(y_1\) 就會從 \(1\) 降到 \(0.9333\)，其他就從 \(0\) 上升到 \(0.0333\)，而不是絕對的 \(0\) 和 \(1\)</p>  
    


## Implement
<p>這邊的 label smoothing 有做一點小修改，因為假設有1個padding的位置，所以均分剩下的 \(\alpha\) 的位置有 \(K-2\)</p>

<p>
$$y_k^{LS} = 
\begin{cases} 
y_k \cdot (1-\alpha), & \text{if } k \text{ is the correct class} \\
\cfrac{\alpha}{K-2}, & \text{otherwise}
\end{cases}$$
</p>


並且是用 KL loss 來算的

$$Loss = \sum_{i=1}^N target(i) \cdot (log(target(i)) - input(i))$$
    
就是因為這個所以要取log再丟進去算

```python
import torch.nn as nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

# Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)  # 5 classes, padding_idx=0, smoothing=0.4
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), # 對每個predict的值取log，因為要用 Kullback-Leibler 散度損失函數
         Variable(torch.LongTensor([2, 1, 0])))  # Targets: class 2, 1, 0

# Show the target distributions expected by the system.
plt.figure(figsize=(8, 6))  # Set figure size for better visualization
plt.imshow(crit.true_dist, cmap='viridis')  # Use 'viridis' colormap for better contrast
plt.colorbar(label='Probability')  # Add colorbar to show probability scale
plt.title('Target Distributions After Label Smoothing (α = 0.4)')
plt.xlabel('Class Index')
plt.ylabel('Sample Index')
plt.xticks(range(crit.size), [f'Class {i}' for i in range(crit.size)])  # Label x-axis with class indices
plt.show()
plt.savefig(os.path.join('label_smoothing.png'))  # Save the plot

print("Loss value:", v.item())  # Print the loss for reference
```

不同步驟填 smooth 後的 predict probability
```python
# 取log
tensor([[   -inf, -1.6094, -0.3567, -2.3026,    -inf],
        [   -inf, -1.6094, -0.3567, -2.3026,    -inf],
        [   -inf, -1.6094, -0.3567, -2.3026,    -inf]])

# 填上 α/K
tensor([[0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        [0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        [0.1333, 0.1333, 0.1333, 0.1333, 0.1333]])

# 填上 confidence
tensor([[0.1333, 0.1333, 0.6000, 0.1333, 0.1333],
        [0.1333, 0.6000, 0.1333, 0.1333, 0.1333],
        [0.6000, 0.1333, 0.1333, 0.1333, 0.1333]])


# 把 padding 的位置填 0
tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        [0.0000, 0.1333, 0.1333, 0.1333, 0.1333]])

```
視覺化後就會看到，第2和第3個類別在不同的sample有最高機率值
![image](https://hackmd.io/_uploads/S14weht9Jx.png)


## KL Loss
根據上面 label smoothing 的方式在是一個
```python
crit = LabelSmoothing(5, 0, 0.1)  # 5 classes, padding_idx=0, smoothing=0.1
def loss(x):
    # 確保 x 為正數，避免數值問題
    x = max(x, 1e-10)  # 避免 x 為 0 或負數
    d = x + 3 * 1  # 計算分母
    # 創建概率分佈，確保總和為 1，並避免 0 值
    predict = torch.FloatTensor([[1e-10, x / d, 1 / d, 1 / d, 1 / d]])
    predict = predict / predict.sum(dim=1, keepdim=True)  # 正規化概率
    # 計算 KL 損失
    return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()

# 繪製損失曲線
x_values = np.arange(1, 100)  # x 從 1 到 99
y_values = [loss(x) for x in x_values]
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='KL Loss with Label Smoothing (α = 0.1)')
plt.xlabel('x')
plt.ylabel('Loss')
plt.title('KL Loss vs. x with Label Smoothing')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig(os.path.join('kl_loss_curve.png'))  # 儲存損失曲線圖表

print("Loss curve generated and saved as 'kl_loss_curve.png'")
```

<p>這裡就要說 label smoothing 的缺點了，如果 \(\alpha\) 太大就會變成 underfit 的問題囉，loss 都完全沒要下降</p>
![image](https://hackmd.io/_uploads/HJRawnFcyx.png)

    
## Example

假設 `size=5`（詞彙表大小為 5），`smoothing=0.1`，`padding_idx=0`：

- 原始 one-hot 標籤（目標為類別 2）：`[0, 0, 1, 0, 0]`
- 平滑後標籤：
    - `confidence = 1 - 0.1 = 0.9`
    - `smoothing / (size - 2) = 0.1 / (5 - 2) = 0.0333`
    - 平滑分佈：`[0, 0.0333, 0.9, 0.0333, 0.0333]`
    - 如果目標是 `padding_idx=0`，則整行設為 0
這種分佈告訴模型：雖然類別 2 是正確答案，但其他類別也有微小的可能性，從而避免模型過於偏向單一預測



## Reference
- [标签平滑（Label Smoothing）详解](https://blog.csdn.net/ytusdc/article/details/128503206)
- [Label Smoothing](https://paperswithcode.com/method/label-smoothing)
- [When Does Label Smoothing Help?](https://arxiv.org/pdf/1906.02629)
    >甚至這篇就有用到蒸餾!

