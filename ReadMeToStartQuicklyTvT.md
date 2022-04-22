### Python environment
Your python must be at least >=3.7, I'm using 3.7.4 currently. If you need more detailed version information about environment, the information is in ``requirements.txt``. 

By the way, compared with last time,  I upgraded some modules to run ``subgraph_mining.Decoder`` :

-  ``torch-geometric``: 1.4.3 -> 2.0.2 
- To install ``torch-geometric`` successfully, you must first upgrade ``torch-scatter`` & ``torch-sparse`` 
- ``torch``: 1.4.0 -> 1.6.0, ``torch-version``: 0.5.0->0.7.0

 

### Dataset & common/data.py
If you want to use new dataset to **train/test model**, change it in ``common/data.py``. You can find the dataset path setting & change the load dataset function in function ``load_dataset()``. If you want to use new dataset to **inference**, change data path in ``subgraph_mining/decoder.py/main()``

We are using ``common/own_dataset_encoder.py`` to load dataset while training/testing, using ``common/own_dataset_decoder.py`` while inference. The DataSource class we mainly use is class ``DiskDataSource``.

P.S. Do not forget to change ``mode = 'Encoder'`` or ``mode = 'Decoder'`` in ``subgraph_matching/config.py`` when you train or inference.



### Run training of encoder(subgraph_matching)
You can train the encoder by ``python -m subgraph_matching.train ``
- model config in ``subgraph_matching/config.py``



### Run test of encoder

1. change the "dataset name" setting in ``subgraph_matching/config.py``:dataset (line 53)
2. ``python -m subgraph_matching.test ``



### Run inference

1. The encoder checkpoint path you are using is in ``subgraph_matching/config.py`` model_path(line 69)
2. The dataset path is in ``subgraph_mining/config.py`` :dataset(line 49)
3. Your pictures are saved in ``./plots/cluster/`` in ``subgraph_mining/decoder.py/pattern_growth()``(line 255)
4. ``python -m subgraph_mining.decoder``



This repo re-uses part of the code from [snap-stanford/neural-subgraph-learning-GNN](https://github.com/snap-stanford/neural-subgraph-learning-GNN)





