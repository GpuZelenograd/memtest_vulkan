#!/bin/bash
GIT_REF_NAME=$1
cargo build --release --target x86_64-pc-windows-gnu --target x86_64-unknown-linux-gnu --target aarch64-unknown-linux-gnu
mkdir target/artifacts
cp -vf target/x86_64-pc-windows-gnu/release/memtest_vulkan.exe target/artifacts/memtest_vulkan-${GIT_REF_NAME}.exe
tar cJvf target/artifacts/DesktopLinux_X86_64-memtest_vulkan-${GIT_REF_NAME}.tar.xz -C target/x86_64-unknown-linux-gnu/release memtest_vulkan
tar cJvf target/artifacts/EmbeddedLinux_AARCH64-memtest_vulkan-${GIT_REF_NAME}.tar.xz -C target/aarch64-unknown-linux-gnu/release memtest_vulkan
