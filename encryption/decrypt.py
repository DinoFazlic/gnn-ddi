#!/usr/bin/env python3
"""
Decrypt an encrypted submission using the RSA private key.

This script is used by the CI pipeline (GitHub Actions) only.
The private key is injected from GitHub Secrets and never stored
in the repository.

Decryption scheme (pure RSA with chunking):
    1. Read the .enc file
    2. Parse [chunk_count] then [len + ciphertext] per chunk
    3. Decrypt each chunk with the RSA private key
    4. Reassemble the original CSV bytes

Usage:
    python encryption/decrypt.py <input_enc> <private_key_pem> <output_csv>

Example (CI only):
    python encryption/decrypt.py \\
        submissions/inbox/team/run_01/predictions.enc \\
        /tmp/private_key.pem \\
        /tmp/predictions.csv
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization


def load_private_key(path: Path):
    """Load an RSA private key from a PEM file."""
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def decrypt_bytes(ciphertext: bytes, private_key) -> bytes:
    """Decrypt data produced by encrypt.encrypt_bytes().

    Input format:
        [4 bytes] number of chunks (big-endian uint32)
        For each chunk:
            [4 bytes] ciphertext length (big-endian uint32)
            [N bytes] ciphertext
    """
    offset = 0

    if len(ciphertext) < 4:
        raise ValueError("Encrypted data is too short (missing chunk count).")

    (num_chunks,) = struct.unpack(">I", ciphertext[offset : offset + 4])
    offset += 4

    plaintext_parts: list[bytes] = []

    for i in range(num_chunks):
        if offset + 4 > len(ciphertext):
            raise ValueError(f"Truncated data: expected chunk {i+1}/{num_chunks} length header.")
        (chunk_len,) = struct.unpack(">I", ciphertext[offset : offset + 4])
        offset += 4

        if offset + chunk_len > len(ciphertext):
            raise ValueError(f"Truncated data: chunk {i+1}/{num_chunks} extends past end of file.")

        chunk_cipher = ciphertext[offset : offset + chunk_len]
        offset += chunk_len

        plaintext_parts.append(
            private_key.decrypt(
                chunk_cipher,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
        )

    return b"".join(plaintext_parts)


def decrypt_file(input_path: Path, key_path: Path, output_path: Path) -> None:
    """Decrypt an .enc file and write the original CSV."""
    private_key = load_private_key(key_path)
    ciphertext = input_path.read_bytes()

    if len(ciphertext) == 0:
        print("❌ Encrypted file is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        plaintext = decrypt_bytes(ciphertext, private_key)
    except Exception as exc:
        print(f"❌ Decryption failed: {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plaintext)

    print(f"✅ Decrypted {input_path} → {output_path}")
    print(f"   Output size: {len(plaintext):,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decrypt an encrypted submission (CI use only)."
    )
    parser.add_argument("input_enc", type=Path, help="Path to the encrypted .enc file")
    parser.add_argument("private_key", type=Path, help="Path to the RSA private key PEM")
    parser.add_argument("output_csv", type=Path, help="Path to write the decrypted CSV")
    args = parser.parse_args()

    if not args.input_enc.exists():
        print(f"❌ Encrypted file not found: {args.input_enc}", file=sys.stderr)
        sys.exit(1)
    if not args.private_key.exists():
        print(f"❌ Private key not found: {args.private_key}", file=sys.stderr)
        sys.exit(1)

    decrypt_file(args.input_enc, args.private_key, args.output_csv)


if __name__ == "__main__":
    main()
