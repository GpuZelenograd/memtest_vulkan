FROM ghcr.io/cross-rs/x86_64-unknown-linux-gnu:main-centos
RUN curl https://sh.rustup.rs -sSf  | sh -s -- -y --default-toolchain 1.64 -t aarch64-unknown-linux-gnu -t x86_64-pc-windows-gnu
RUN yum -y install epel-release
RUN yum -y install gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu
RUN rpm -i https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw-binutils-generic-2.25-1.el7.x86_64.rpm https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw64-binutils-2.25-1.el7.x86_64.rpm https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw64-filesystem-101-1.el7.noarch.rpm https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw-filesystem-base-101-1.el7.noarch.rpm
RUN rpm -i https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw64-cpp-4.9.3-1.el7.x86_64.rpm https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw64-crt-4.0.4-3.el7.noarch.rpm https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw64-headers-4.0.4-5.el7.noarch.rpm https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw64-winpthreads-4.0.4-1.el7.noarch.rpm
RUN rpm -i https://dl.fedoraproject.org/pub/archive/epel/7.2019-05-29/x86_64/Packages/m/mingw64-gcc-4.9.3-1.el7.x86_64.rpm

ENTRYPOINT ["/github/workspace/.github/actions/build-in-docker/build-in-docker.sh"]
