FROM ghcr.io/cross-rs/x86_64-unknown-linux-gnu:main-centos
RUN curl https://sh.rustup.rs -sSf  | sh -s -- -y --default-toolchain 1.64 -t aarch64-unknown-linux-gnu -t x86_64-pc-windows-gnu
COPY build-in-docker.sh /build-in-docker.sh
ENTRYPOINT ["/build-in-docker.sh"]
