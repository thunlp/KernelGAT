# Abstract Reranking in SCIKGAT

For the abstract reranking module, we follow the [previous work](https://arxiv.org/abs/2005.02365) and fine-tune our in-domain language model (MLM trained with CORD data) with the medical corpus from MS-MARCO to fit our abstract retrieval module to the open-domain COVID related literature search. More details can refer to our [paper](https://arxiv.org/abs/2011.01580) and [OpenMatch](https://github.com/thunlp/OpenMatch) toolkit to achieve top retrieval for COVID search.
