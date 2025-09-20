#!/bin/bash
(
    set -xeu
    cargo fmt --check --all
    cargo clippy --all -- -D clippy::all -W clippy::too-many-arguments
    RUSTFLAGS="-D warnings" cargo check --all-features --all-targets
)
