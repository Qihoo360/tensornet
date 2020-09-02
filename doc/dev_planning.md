# 反馈及开发计划

- [ ] 同步版支持。
    在机器带宽10G以上，使用gpu的集群中，预计同步版的性能会更好。同步版可以避免异步版本局部变量无法同步的问题，比如BN中的移动平均，胶囊网络中的动态路由参数等等。

- [ ] ​使用blocking queue优化sparse push动作性能。
    预计可以将sparse push的耗时减少一半。

- [ ] 更多模型示例支持。
    编写更多版本的模型，拷贝即使用。

- [ ] 节点数变更支持。
    目前使用增量训练只能使用与全量训练相同的节点数进行训练，节点数变化之后需要将保存的模型重新分布式切分到不同的节点上。

- [ ] MLIR inference支持。
    MLIR作为长远计划，配合TensorNet保存的模型可以通过定制dialect优化inference embedding生成的相关逻辑。

- [ ] EmbeddingFeatures layer支持不同dimension的feature column。
    目前由于最初设计EmbeddingFeatures中只支持相同dimension的column，近期反馈支持不同dimension的需求比较多，有必要支持。

- [ ] 指标在所有节点之间聚合。
    目前训练结果的指标是每个节点单独计算的，如果计算一个所有节点的指标会更好，但是在异步模式下，不好更改TF自身的代码，开发有点困难，不过当训练数据充分的时候每个节点的指标相差不会很大。

- [ ] FTRL优化器支持。

