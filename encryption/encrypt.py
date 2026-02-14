#!/usr/bin/env python3
"""
Encrypt a submission CSV using the competition's RSA public key.

Encryption scheme (pure RSA with chunking):
    1. Read the CSV file as raw bytes
    2. Split into chunks of ≤190 bytes (safe for RSA-2048 with OAEP/SHA-256)
    3. Encrypt each chunk independently with the RSA public key
    4. Write output as: [4-byte big-endian chunk count] + [4-byte big-endian len + ciphertext] per chunk

Usage:
    python encryption/encrypt.py <input_csv> <public_key_pem> <output_enc>

Example:
    python encryption/encrypt.py \\
        submissions/inbox/my_team/run_01/predictions.csv \\
        encryption/public_key.pem \\
        submissions/inbox/my_team/run_01/predictions.enc
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization


# RSA-2048 with OAEP SHA-256 can encrypt at most 190 bytes per chunk
# (256 - 2*32 - 2 = 190 bytes)
_MAX_CHUNK_SIZE = 190


def load_public_key(path: Path):
    """Load an RSA public key from a PEM file."""
    return serialization.load_pem_public_key(path.read_bytes())


def encrypt_bytes(plaintext: bytes, public_key) -> bytes:
    """Encrypt arbitrary-length plaintext using RSA with chunking.

    Output format:
        [4 bytes] number of chunks (big-endian uint32)
        For each chunk:
            [4 bytes] ciphertext length (big-endian uint32)
            [N bytes] ciphertext
    """
    chunks = [
        plaintext[i : i + _MAX_CHUNK_SIZE]
        for i in range(0, len(plaintext), _MAX_CHUNK_SIZE)
    ]

    parts: list[bytes] = []
    parts.append(struct.pack(">I", len(chunks)))

    for chunk in chunks:
        ciphertext = public_key.encrypt(
            chunk,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        parts.append(struct.pack(">I", len(ciphertext)))
        parts.append(ciphertext)

    return b"".join(parts)


def encrypt_file(input_path: Path, key_path: Path, output_path: Path) -> None:
    """Encrypt a file and write the result."""
    public_key = load_public_key(key_path)
    plaintext = input_path.read_bytes()

    if len(plaintext) == 0:
        print("❌ Input file is empty.", file=sys.stderr)
        sys.exit(1)

    ciphertext = encrypt_bytes(plaintext, public_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(ciphertext)

    print(f"✅ Encrypted {input_path} → {output_path}")
    print(f"   Input size:  {len(plaintext):,} bytes")
    print(f"   Output size: {len(ciphertext):,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt a submission CSV with the competition RSA public key."
    )
    parser.add_argument("input_csv", type=Path, help="Path to your predictions CSV")
    parser.add_argument(
        "public_key",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "public_key.pem",
        help="Path to the RSA public key (default: encryption/public_key.pem)",
    )
    parser.add_argument(
        "output_enc",
        type=Path,
        nargs="?",
        default=None,
        help="Output path for the encrypted file (default: <input>.enc)",
    )
    args = parser.parse_args()

    if args.output_enc is None:
        args.output_enc = args.input_csv.with_suffix(".enc")

    if not args.input_csv.exists():
        print(f"❌ Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    if not args.public_key.exists():
        print(f"❌ Public key not found: {args.public_key}", file=sys.stderr)
        sys.exit(1)

    encrypt_file(args.input_csv, args.public_key, args.output_enc)


if __name__ == "__main__":
    main()
