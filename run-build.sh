#!/bin/bash
(
    # getting all below targets to link may be a bit tricky, especially x86_64-win7-windows-gnu,
    # so prepared Docker builder image provided at https://github.com/galkinvv/manycross2014/pkgs/container/manycross2014
    set -xeu
    GIT_REF_NAME="${1:-${GITHUB_REF_NAME:-`date +%Y-%m-%dT%H-%M-%S`}}"
    GIT_REF_NAME="${GIT_REF_NAME/\//---}"
    cargo build --release --target x86_64-pc-windows-gnu --target x86_64-unknown-linux-gnu --target aarch64-unknown-linux-gnu
    RUSTC_BOOTSTRAP=1 cargo build -Zbuild-std --release --target x86_64-win7-windows-gnu
    mkdir -p target/artifacts
    cp -vf target/x86_64-pc-windows-gnu/release/memtest_vulkan.exe target/artifacts/memtest_vulkan-${GIT_REF_NAME}-alt-experimental.exe
    cp -vf target/x86_64-win7-windows-gnu/release/memtest_vulkan.exe target/artifacts/memtest_vulkan-${GIT_REF_NAME}.exe
    tar cJvf target/artifacts/memtest_vulkan-${GIT_REF_NAME}_DesktopLinux_X86_64.tar.xz -C target/x86_64-unknown-linux-gnu/release memtest_vulkan
    tar cJvf target/artifacts/memtest_vulkan-${GIT_REF_NAME}_EmbeddedLinux_AARCH64.tar.xz -C target/aarch64-unknown-linux-gnu/release memtest_vulkan
)
