# HIN-Align

Bachelor degree research on heterogeneous information network alignment. Extended work of [GCN-Align](https://github.com/1049451037/GCN-Align).

Datasets are from [JAPE](https://github.com/nju-websoft/JAPE) and [IONE](https://github.com/ColaLL/IONE).

# Environment

* python>=3.5
* tensorflow>=1.10.1
* scipy>=1.1.0
* networkx>=2.2

# Test

```
unzip dbp15k.zip
chmod +x test.sh
./test.sh
```

The pre-trained results are in the *res/* folder. If you don't want to train by yourself, just see the files in it.

For social network, run:

```
python train_sn.py --seed 5
```

For automatically train weights:

```
python train_auto.py
```

# Citation

Please politely cite our work as follows:

*Zhichun Wang, Qingsong Lv, Xiaohan Lan, Yu Zhang. Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks. In: EMNLP 2018.*

# TODO

- [x] Change a\_ij to sigmoid(a\_ij)
- [x] Combine with TransE (KG) or DeepWalk (SN)
- [ ] Combine with MT
- [x] Social Network Alignment
- [ ] Iterative or Bootstrapping
- [ ] Use [faiss](https://github.com/facebookresearch/faiss) to improve evaluation speed
- [ ] Dimension Reduction or other ways of combination
- [x] Automatic training for hybrid weights
- [ ] Batched training for GCN
- [ ] Try other GNN models
