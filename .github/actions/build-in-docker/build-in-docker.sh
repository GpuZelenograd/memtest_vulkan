#!/bin/bash
export HOME=/root
export PATH=$HOME/.cargo/bin:$PATH
rustup default 1.64.0
cargo --list
rustup target list
cargo build --release --target x86_64-pc-windows-gnu
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
