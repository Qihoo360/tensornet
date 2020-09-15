# 编译与部署

## clone代码

    git clone git@adgit.src.corp.qihoo.net:zhangyansheng/tensornet.git

## 准备环境
1. 安装anaconda3。

2. 安装tensorflow。目前只支持2.2.0，其它版本陆续会适配。

    ```
    pip install tensorflow==2.2.0
    ```

3. 安装bazel。

    bazel的版本最好使用2.2.0，tensorflow每个版本使用的bazel版本不一样，具体版本请看：[https://github.com/tensorflow/tensorflow/blob/master/.bazelversion](https://github.com/tensorflow/tensorflow/blob/master/.bazelversion)。
    bazel的安装最好使用github中已经release预编译好的binary，具体请参考：[https://docs.bazel.build/versions/master/install-os-x.html](https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu)

4. 安装Open MPI。

    tensornet使用MPI做集群管理。我们推荐使用OpenMPI，最好使用源码安装，安装方法见[https://www.open-mpi.org/faq/?category=building#easy-build](https://www.open-mpi.org/faq/?category=building#easy-build)。OpenMpi版本的应该与集群内每台机器上部署的版本一致，我们内部使用的一个较老的版本1.4.5。

    ```
    wget https://download.open-mpi.org/release/open-mpi/v1.4/openmpi-1.4.5.tar.gz
    tar -zxvf openmpi-1.4.5.tar.gz
    ./configure CFLAGS="-fPIC" CXXFlAGS="-fPIC" --prefix=/da2/zhangyansheng/openmpi-1.4.5 --enable-static
    make -j20
    make install
    ```

## 编译

执行下面命令编译：

    sh ./configure.sh --openmpi_path /da2/zhangyansheng/openmpi-1.4.5
    bazel build -c opt //core:_pywrap_tn

## 部署

编译完成之后的so在`bazal-bin/core`目录下，其拷贝到python包所在的目录就可以随意使用了。

    cp -f bazel-bin/core/_pywrap_tn.so tensornet/core 
    export PYTHONPATH=$(pwd):${PYTHONPATH}
    export LD_LIBRARY_PATH="/da2/zhangyansheng/openmpi-1.4.5/lib:${LD_LIBRARY_PATH}"
    python -c "import tensorflow as tf; import tensornet as tn; print(tn.version)"

在`tools`目录下我们提供了一个脚本`sh tools/package.sh`可以将python程序及依赖的so打包起来，用户在使用的时候只需要将这个包拷贝到自己的项目下面，通过设置PYTHONPATH就可以导入使用了。

## tips

### bazel build的时候下载依赖包比较慢如何解决？

方案一：
1. 在`WORKSPACE`文件中找到需要下载的包的http路径
2. 可以直接改成从本地文件读取
3. 比如对于tensorflow将`https://github.com/tensorflow/tensorflow/archive/v2.2.0.tar.gz`改成`file:///da2/zhangyansheng/package/tensorflow-v2.2.0.tar.gz`，这样可以使用迅雷等下载工具下载好之后再build。

方案二：
1. `bazel build`的过程当中会打印下载包的http链接
2. 使用下载工具下载
3. 下载完这个包之后计算sha256，比如tensorflow包：`cat tensorflow-2.2.0.tar.gz | sha256sum`为`69cd836f87b8c53506c4f706f655d423270f5a563b76dc1cfa60fbc3184185a3`。
4. 进入`~/.cache/bazel/_bazel_zhangyansheng/cache/repos/v1/content_addressable/sha256`目录，中间的username改成自己的username
5. 创建以sha256为名从目录。`mkdir 69cd836f87b8c53506c4f706f655d423270f5a563b76dc1cfa60fbc3184185a3`
6. 将下载的包mv到这个目录下，名字为`file`。`mv /da2/zhangyansheng/package/tensorflow-v2.2.0.tar.gz file`
7. 重新build会自动使用cache目录下的包。

### gcc版本如何选择。

gcc一定要支持到c++14，我们内部使用的是GCC 5.5.8

