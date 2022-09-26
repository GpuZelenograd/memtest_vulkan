#!/bin/bash
id
pwd
ls
set
source "$HOME/.cargo/env"
cargo --list
rustup target list
cargo build --release --target x86_64-pc-windows-gnu
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
