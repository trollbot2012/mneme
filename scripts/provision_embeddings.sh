#!/bin/sh
# Provision the ADR-0004 embedding encoder artifacts (all-MiniLM-L6-v2 quint8
# avx2 ONNX export + vocab.txt) into a models/ directory.
#
# CHECK-THEN-SKIP: a file already present with a matching pinned sha256 is
# never re-downloaded — a provisioned box exits 0 with zero network traffic.
# Downloads stage to <name>.part and rename atomically; any sha256 mismatch
# deletes the downloaded copy and fails loud.
#
# This script is the ONLY place network provisioning happens: mneme.py ships
# zero network code, and the optional onnxruntime wheel is NEVER pip-installed
# here (see the hint printed at the end).
#
# The URL pins the upstream revision matching ENCODER_ID (rev1110a243); the
# pinned sha256 below — taken from the load-verified deployment artifact — is
# the actual integrity gate for both files.
#
# Usage:
#   sh scripts/provision_embeddings.sh [--dest DIR]
# Default DIR: $MNEME_HOME/models (else ~/.mneme/models)

set -eu

DEST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --dest=*) DEST="${1#--dest=}"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$DEST" ]; then
    DEST="${MNEME_HOME:-$HOME/.mneme}/models"
fi

REV="1110a243"
BASE="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$REV"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "need sha256sum or shasum on PATH" >&2
        exit 3
    fi
}

mkdir -p "$DEST"

fetch_pinned() {
    name="$1"; url="$2"; bytes="$3"; want="$4"
    target="$DEST/$name"
    if [ -f "$target" ]; then
        if [ "$(sha256_of "$target")" = "$want" ]; then
            echo "OK (already provisioned, sha256 verified): $target"
            return 0    # check-then-skip: no network for present, verified files
        fi
        echo "WARNING: sha256 mismatch on existing $name — deleting and refetching" >&2
        rm -f "$target"
    fi
    part="$target.part"
    rm -f "$part"
    echo "downloading $url -> $target ($bytes bytes)"
    curl -fL --retry 3 -o "$part" "$url"
    got="$(sha256_of "$part")"
    if [ "$got" != "$want" ]; then
        rm -f "$part"   # delete on mismatch, fail loud
        echo "sha256 mismatch for $name: expected $want, got $got — downloaded copy deleted" >&2
        exit 4
    fi
    mv -f "$part" "$target"   # atomic-rename publish
    echo "OK (downloaded, sha256 verified): $target"
}

fetch_pinned "model_quint8_avx2.onnx" "$BASE/onnx/model_quint8_avx2.onnx" \
    23046789 "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"
fetch_pinned "vocab.txt" "$BASE/vocab.txt" \
    231508 "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"

echo ""
echo "models ready in: $DEST"
echo "Embeddings also need the optional onnxruntime wheel (NOT installed by this script):"
echo '    pip install "onnxruntime>=1.20"        # or: pip install "mneme-memory[embeddings]"'
echo "Point Mneme at the directory via the 'embed_model_dir' config key,"
echo "or place models/ next to the store's mneme.db — 'auto' does the rest."
