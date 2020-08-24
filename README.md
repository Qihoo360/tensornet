![TensorNet](doc/logo.png)

**TensorNet**是一个构建在**TensorFlow**之上针对广告推荐等大规模稀疏场景优化的分布式训练框架。TensorNet的目标是让所有使用TensorFlow的开发者可以快速的、方便的训练出稀疏参数超过百亿的超大模型。

## TensorNet主要改进

在广告、搜索、推荐等场景下的深度模型都会面临样本规模非常大，参数非常多的挑战，对于这样的模型必须要采用分布式的方式训练。TensorNet主要对TensorFlow的下面几个核心点进行优化：

### 1. 对sparse embedding优化。

在使用TensorFlow构造带有稀疏特征的模型的时候必须要构造一个大的**embedding tensor**，这个embedding tensor通常非常大，比如对于用户维度的embedding tensor一般会超过亿维，当在这么大的tensor之上做查找、同步等操作会带来巨大的性能开销，严重拖慢分布式训练的速度。

TensorNet使用一个**较小的**、能够容纳**一个batch**数据的tensor代替原始的embedding tensor，将sparse embedding tensor的查找和同步开销优化到最小。

### 2. 对参数服务器优化。

在TensorFlow 1.x中官方实现的参数服务器参数按照tensor分隔，容易导致某一节点变成热点，另外对参数服务器的参数分配工作比较繁琐。在TensorFlow 2.x的版本中官方倾向于优化同步训练模式，对参数服务器的支持越来越少。

TensorNet按照sparse feature key的**哈希值**均匀分隔参数到每个节点，极大的避免了热点问题，速度得到了保障。并且分配参数工作对开发者透明，上手较快。

## 文档

1. [编译与部署](doc/compile_deploy.md)
2. [quick start with wide deep](doc/tutorial/01-begin-with-wide-deep.ipynb)
3. [在集群上运行](doc/tutorial/02-run-in-cluster.ipynb)

## License

[Apache License 2.0](LICENSE)

## concat us

QQ群号：1146192156

![TensorNet](doc/TensorNet-QR-code.png)
