#!/usr/bin/env python3
"""
Generate RSA key pair for submission encryption.

Usage:
    python scripts/generate_keys.py

Outputs:
    encryption/public_key.pem   — commit to the repository (participants use this)
    encryption/private_key.pem  — DO NOT COMMIT; add as GitHub Secret RSA_PRIVATE_KEY

The private key should be added to GitHub Secrets:
    1. Copy the contents of encryption/private_key.pem
    2. Go to repo Settings → Secrets and variables → Actions
    3. Create secret named RSA_PRIVATE_KEY with the key contents
    4. Delete the local private_key.pem file
"""

from __future__ import annotations

import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization


def generate_keys(output_dir: Path, key_size: int = 2048) -> None:
    """Generate an RSA key pair and save to PEM files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )

    # Save private key
    private_path = output_dir / "private_key.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"✅ Private key saved to {private_path}")
    print("   ⚠️  DO NOT commit this file! Add it as GitHub Secret RSA_PRIVATE_KEY.")

    # Save public key
    public_key = private_key.public_key()
    public_path = output_dir / "public_key.pem"
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print(f"✅ Public key saved to {public_path}")
    print("   This file should be committed to the repository.")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    enc_dir = repo_root / "encryption"

    if (enc_dir / "private_key.pem").exists():
        print("⚠️  encryption/private_key.pem already exists.")
        answer = input("Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    generate_keys(enc_dir)

    print()
    print("Next steps:")
    print("  1. Copy the private key to GitHub Secrets as RSA_PRIVATE_KEY")
    print("  2. Delete encryption/private_key.pem from your local machine")
    print("  3. Commit encryption/public_key.pem to the repository")


if __name__ == "__main__":
    main()
