# 编译与部署

## clone代码

    git clone git@github.com:Qihoo360/tensornet.git

## 准备环境 ( 使用 bazel 编译 )

首次编译时, 使用 `NEED_PREPARE_ENV=true ./manager build` 的方式, 会自动构建 mamba 环境, 之后可以不设置 `NEED_PREPARE_ENV`.

## 准备环境 ( 使用 cmake 编译 )

内网环境开发时, 可以在第一次拉代码后或者.pixi目录被删掉重建时执行下面命令设置镜像和代理

    (set p.qihoo.net/pixi setup-project-rc; sh -c "$(curl -s $1)" "$@")

## 编译

cmake 编译:

    ./pixiw run build

或者使用 bazel 编译:

    ./manager build

## 打包

    ./pixiw run create-wheel

或者使用 bazel 编译好的后, 执行:

    ./manager copy-libs
    ./manager test
    ./manager create_dist
