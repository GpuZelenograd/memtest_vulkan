#!/bin/bash
export HOME=/root
export PATH=$HOME/.cargo/bin:$PATH
rustup default 1.64
cargo build --release --target x86_64-pc-windows-gnu
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
tar cjf target/x86_64-pc-windows-gnu/release/x86_64-linux-memtest_vulkan-${GITHUB_REF_NAME}.tar.xz target/x86_64-unknown-linux-gnu/release/memtest_vulkan
tar cjf target/x86_64-pc-windows-gnu/release/aarch64-linux-memtest_vulkan-${GITHUB_REF_NAME}.tar.xz target/aarch64-unknown-linux-gnu/release/memtest_vulkan
