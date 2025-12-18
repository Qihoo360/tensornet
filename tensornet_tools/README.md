## 合并sparse file

由于生成的sparse文件分布在各个目录, 可以通过spark将sparse文件结果抽取结果并合并到一个目录下

```bash
spark-submit3 --executor-memory 8g --driver-memory 10g --py-files utils.py merge_sparse.py -i "/user/test/model_path/sparse_table/*/*/*bin.gz" -o "/user/test/model_merge_path" -f 'bin' -n 500 -b
```

### 参数配置

配置名称 | 默认值 | 含义
----------- | ----------- | -----------
-i/--input  |  None  |  输入路径
-o/--output |  None  |  输出路径
-f/--format |  bin   |  输入文件格式
-n/--number |  20    |  输出并行度
-b/--bracker | False |  输出的Weights是否需要用[]包括, []当作一列, 用\t分割


## sparse切换并行度

现阶段生成的sparse_table目录并行度无法切换，如果前后不一致会导致数据缺失问题，无法扩缩容。通过spark读入原始数据，按照指定的并行度输出文件pattern

由于使用了hdfs3来写入文件，需要打包上传环境，使用[env文件](config/tn_tool_env.yaml)

```bash
spark-submit3 --conf spark.executor.memory=10g --conf spark.archives=hdfs://nn/user/test/cache/python.tar.gz#envs --conf spark.pyspark.driver.python=/home/test/micromamba/envs/tn_tool_env/bin/python --conf spark.pyspark.python=./envs/bin/python  --py-files utils.py resize_sparse.py --input /user/test/model/* --output /user/test/resize --number 50
```

### 参数配置

配置名称 | 默认值 | 含义
----------- | ----------- | -----------
-i/--input  |  None  |  输入路径, 会抓取hdfs头用作hdfs文件写入，如没有hdfs头会默认用hdfs://ss-hadoop2
-o/--output |  None  |  输出路径,会在输出路径下生成 handle_name/rank_number/block_num.gz 文件
-f/--format |  bin   |  输入文件格式
-n/--number |  20    |  输出并行度


## dense切换并行度

和 sparse 类似

```bash
spark-submit3 --conf spark.executor.memory=10g --conf spark.archives=hdfs://nn/user/test/cache/python.tar.gz#envs --conf spark.pyspark.driver.python=/home/test/micromamba/envs/tn_tool_env/bin/python --conf spark.pyspark.python=./envs/bin/python  --py-files utils.py resize_dense.py --input /user/test/model/* --output /user/test/resize --number 50
```

### 参数配置

配置名称 | 默认值 | 含义
----------- | ----------- | -----------
-i/--input  |  None  |  输入路径, 会抓取hdfs头用作hdfs文件写入，如没有hdfs头会默认用hdfs://ss-hadoop2
-o/--output |  None  |  输出路径,会在输出路径下生成 handle_name/rank_number 文件
-n/--number |  20    |  输出并行度
