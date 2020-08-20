# TensorNet

TensorFlow自身实现的分布式方案在数据规模较小，节点数较少的场景下较为友好，但是在数据规模庞大、节点数较多的场景下其运行速度比较慢。

主要原因为：

1. TensorFlow使用同步方案。同步训练过程中因为慢节点的存在，使得训练速度依赖于最慢节点的执行速度。当节点数变多时，出现慢节点的概率越大，训练速度也越慢。
2. TensorFlow会对模型中的所有参数做all reduce，尤其对于sparse embedding数据较大，同步过程耗时较长。比如对于一个shape为(1000000, 8)占用内存30M的embedding，其必须将所有数据同步到其它节点，造成带宽不必要的浪费，严重降低了执行效率。
3. TensorFlow支持的embedding维度不易过大。当embedding的大小设置的较大的时候不仅会占用较多内存，而且会降低embedding_lookup的速度。


tensornet针对上面这些问题，主要对embedding层进行优化，减少通信数据量，提升执行速度。

1. 使用异步方案。每个batch训练过程中没有必要等待慢节点。
2. tensornet sparse embedding数据不会做allreduce，每个节点只拉取自身相关的数据。
3. 使用内部的mapping 方法减小embedding层variable的大小。TensorFlow中会对一个特征预先设定一个大的shape以创建其对应的Variable，比如对于有1亿个维度的userid特征，会创建一个维度为1亿的variable，而对于每个batch训练过程当中，只会用到其中很少维度的权重，那么我们将只在每个batch训练过程当中用到的维度挑出来，并mapping到新的embedding id上，拼凑成新的variable，从而将embedding层variable的大小减小到最小。


## 编译

可以使用release预编译好的版本。如果需要编译请参考[编译与部署](doc/compile_deploy.md)
