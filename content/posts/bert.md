---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
date: 2024-12-08T23:18:01+08:00
summary: "論文引用: Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. North American Chapter of the Association for Computational Linguistics."
tags: ["bert"]
---

論文引用: Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. North American Chapter of the Association for Computational Linguistics.


因為GPT 1.0的發表，Google決定乘勝追擊，在2019推出BERT這個語言模型，相較於ELMo和GPT的下游單向的訓練方式，BERT用了雙向的Encoder，讓每個節點的到的上下文(context)資訊增加，當時的表現也是在多項語料庫上超越GPT1.0

<br>

# Pre-traning Tasks
## Task 1. Masked LM

- Why
- How
- 會遮蓋掉15%的token，遮蓋掉的部分會用特殊的`[MASK]`符號取代，模型只會關注
根據作者在論文中提到的預訓練任務畫成圖大概長以下這樣


<img src="base_structure.png" height=100 width =1000 style=" margin: auto; display: block;">


## Task 2. Next Sentence Prediction (NSP)


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

