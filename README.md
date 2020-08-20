# TensorNet

在广告、搜索、推荐等场景下的深度模型都会存在大量的高维离散稀疏特征，这些模型通常都需要一个大的embedding矩阵，模型训练的时候需要从这个embedding矩阵中查找具体特征的embedding值，在embedding值之上构造不同结构的模型，比如最典型的wide_deep模型。训练这样一个模型有两个基本问题：

1. 训练样本规模非常大，单机训练速度慢。比如对于日PV为亿级的推荐系统一天会产生上亿条训练样本。
2. 样本参数维度大，单机装不下。比如对于有50亿参数的wide_deep模型需要近150G内存存储权重信息（embedding size为8，按float 32存储，不包括不同优化器上的超参占用的内存）；

一般开发者都会选择TensorFlow作为模型的训练框架，但自TensorFlow 2.0之后官方实现中对参数服务器的支持越来越少，其实现的分布式方案在数据规模较小，节点数较少的场景下较为友好，在数据规模庞大、节点数较多的场景下其运行速度比较慢。

主要原因为：

1. TensorFlow 2.x使用同步方案。同步训练过程中因为慢节点的存在，使得训练速度依赖于最慢节点的执行速度。当节点数变多时，出现慢节点的概率越大，训练速度也越慢。
2. TensorFlow 2.x会对模型中的所有参数做all reduce，尤其对于sparse embedding数据较大，同步过程耗时较长，造成带宽不必要的浪费，严重降低了执行效率。
3. TensorFlow 2.x支持的embedding维度不易过大。当embedding的大小设置的较大的时候不仅会占用较多内存，而且会降低embedding_lookup的速度。


我们针对TensorFlow 2.x所存在的问题，在TensorFlow基础之上，二次开发了一套可以支持高维稀疏特征的训练框架——tensornet：

1. tensornet针对embedding层进行优化，减少通信数据量，提升执行速度。
2. tensornet使用异步方案。每个batch训练过程中没有必要等待慢节点。
3. tensornet sparse embedding数据不会做allreduce，每个节点只拉取自身相关的数据。


在360广告真实场景下，训练样本过百亿，参数几十亿，使用TensorFlow基本不能完成训练，tensornet完成了从无到有的跨越，在使用CPU搭建的HPC集群（网卡速度统一为10GbE）上我们测试的性能结果如下：

| 节点数 | 单节点core size | 样本量 | 训练速度 |
| ------ | --------------- | ------ | -------- |
| 50     | 5               | 3000w  | 25min    |

## 编译

可以使用release预编译好的版本。如果需要编译请参考[编译与部署](doc/compile_deploy.md)

## tutorial

1. [quick-start](doc/tutorial/01-quick-start.md)
2. [split-sub-graph](doc/tutorial/01-quick-start.md)
3. [inference](doc/tutorial/01-quick-start.md)


## License

[Apache License 2.0](LICENSE)
