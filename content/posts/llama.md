---
title: "LLaMA: Open and Efficient Foundation Language Models"
date: 2024-12-08T21:55:50+08:00
summary: "論文引用: Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., & Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models. ArXiv, abs/2302.13971."
tags: ["llama"]
---

論文引用: Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., & Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models. ArXiv, abs/2302.13971.
<img src="llama_milestone.png" width =1000 style=" margin: auto; display: block;">
<br>

# Challenge Scaling Law
## Scaling Law
先說甚麼是Scaling Law

關於更詳細的Scaling Law可以參考這篇[論文](https://arxiv.org/abs/2001.08361)
## 羊駝的覺醒
- 論文中提到: **LLM with fast inference rather than a fast training process**，以前會考慮到scaling law是因為想要訓練的時間短一點，但是訓練時間短對於LLM的使用並沒有幫助，我們想要的是在使用LLM時可以更快速的得到想要的回答 -- 也就是在inference時快一點，在訓練時慢一點沒差

- 那要怎麼讓參數小於GPT 十倍之多的llama 1.0有較好的表現呢?就是給他訓練資料多一點，訓練時常久一點，即使是小模型也能在多次訓練後有較好的表現!
<img src="scaling_law.png" height=100 width=1000 style=" margin: auto; display: block;">


# Results
- 雖然參數少很多，但是在許多與料庫上的表現都優於GPT
<img src="results_1.png" height=100 width=1000 style=" margin: auto; display: block;">
<img src="results_2.png" height=100 width=1000 style=" margin: auto; display: block;">

