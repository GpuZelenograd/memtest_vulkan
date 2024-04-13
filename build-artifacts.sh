#!/bin/bash
(
    set -xe
    GIT_REF_NAME="${1:-${GITHUB_REF_NAME/\//---}}"
    cargo build --release --target x86_64-pc-windows-gnu --target x86_64-unknown-linux-gnu --target aarch64-unknown-linux-gnu
    mkdir -p target/artifacts
    cp -vf target/x86_64-pc-windows-gnu/release/memtest_vulkan.exe target/artifacts/memtest_vulkan-${GIT_REF_NAME}.exe
    tar cJvf target/artifacts/memtest_vulkan-${GIT_REF_NAME}_DesktopLinux_X86_64.tar.xz -C target/x86_64-unknown-linux-gnu/release memtest_vulkan
    tar cJvf target/artifacts/memtest_vulkan-${GIT_REF_NAME}_EmbeddedLinux_AARCH64.tar.xz -C target/aarch64-unknown-linux-gnu/release memtest_vulkan
)
