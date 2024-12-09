---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
date: 2024-12-08T23:18:01+08:00
summary: "論文引用: Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. North American Chapter of the Association for Computational Linguistics."
tags: ["bert"]
---

論文引用: Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. North American Chapter of the Association for Computational Linguistics.


因為GPT 1.0的發表，Google決定乘勝追擊，在2019推出BERT這個語言模型，相較於ELMo和GPT的下游單向的訓練方式，BERT用了雙向的Encoder，讓每個節點的到的上下文(context)資訊增加，當時的表現也是在多項語料庫上超越GPT1.0

<br>

# Contextualized  Embeddings
同樣一個詞在不同語境下意義就會不同，所以比起以前的word vector一個蘿蔔一個坑，現在大家更關心的是如何量化前後文讓模型更能推敲出一個詞在不同語境的意思

tbc...


<br>


# Pre-traning Tasks
用unlabed data(未標記、沒答案的資料)來訓練模型，未下游任務找到一個較好的初始點
## Task 1. Masked LM

- Why
    >Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.

    因為BERT用的是雙向的Encoder，這樣一來一個節點不就後面前面是啥都知道了嗎?對於QA這種task不就無法用了嗎?

    所以，為了避免模型對於前面後面的context(上下文)搞混，用這種填充的訓練方式增強模型的了解文本的能力，這也是作者從[克漏字](https://gwern.net/doc/psychology/writing/1953-taylor.pdf)得到的啟發，就是這麼神奇

- How\
    會遮蓋掉15%的token，遮蓋掉的部分會用特殊的`[MASK]`符號取代，模型只會關注被遮蓋的位置，經過12個encoder後，最後送到Softmax過濾，看哪個詞的機率最高的就是模型預測應該要放的詞

    根據作者在論文中提到的BERT base(基礎版)預訓練任務畫成圖大概長以下這樣

    <img src="base_structure.png" height=100 width =800 >


## Task 2. Next Sentence Prediction (NSP)
就是字面上的意思，因為下游任務很多這種給模型一個句子，然後要模型分辨是正負面、entailment(文本大意)、similarity(相似度)等，為了在finetuned時有更好的表現，先用這個任務讓模型熟悉之後要做的事
- 我也是沒想到模型就這麼聽話，真的比沒有NSP這個預訓練任務的模型表現好欸\
    可以來看一下作者們做的消融實驗(ablatoin study)表格中，`LTR & No NSP` 是left-to-right並且沒有NSP預訓練任務的模型(感覺就是在說GPT 1.0)，然後 `BiLSTM` 雙向的LSTM就很像在說ELMo，總而言之就是各種跟別人的比較(要凸顯自己很強)
    <img src="ablation_study_nsp.png" height=100 width =500 style="display: block;">



<br>


# Reference
- [ELMo算法详解](https://blog.csdn.net/qq_42791848/article/details/122374703)
- [Learn how to build powerful contextual word embeddings with ELMo](https://medium.com/saarthi-ai/elmo-for-contextual-word-embedding-for-text-classification-24c9693b0045)
- [浅谈feature-based 和 fine-tune](https://blog.csdn.net/weixin_46707326/article/details/123451774)
- [CoVe GitHub](https://github.com/salesforce/cove)
- [ELMo 一词多义](https://mofanpy.com/tutorials/machine-learning/nlp/elmo)
- [31. ELMo (Embeddings from Language Models 嵌入式語言模型)](https://medium.com/programming-with-data/31-elmo-embeddings-from-language-models-%E5%B5%8C%E5%85%A5%E5%BC%8F%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8B-c59937da83af)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [BERT Model – Bidirectional Encoder Representations from Transformers](https://quantpedia.com/bert-model-bidirectional-encoder-representations-from-transformers/)
- [BERT: State-of-the-Art Model for Natural Language Processing](https://www.comet.com/site/blog/bert-state-of-the-art-model-for-natural-language-processing/)

