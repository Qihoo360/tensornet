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
