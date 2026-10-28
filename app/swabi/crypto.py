"""Decrypt CryptoJS AES-encrypted Swabi backend responses.

Format produced by CryptoJS.AES.encrypt(plaintext, passphrase):
    base64( b"Salted__" + salt[8] + AES-CBC-ciphertext )

Key + IV derived via EVP_BytesToKey:
    D_1 = H(passphrase + salt)
    D_2 = H(D_1 + passphrase + salt)
    D_3 = H(D_2 + passphrase + salt)
    key = (D_1 || D_2)[:32]   # 32-byte AES-256 key
    iv  = D_3[:16]            # 16-byte CBC IV

H = MD5 (CryptoJS default) or SHA256 if the frontend was configured
that way. Try MD5 first; switch to SHA256 if decryption produces
garbage (PaddingError or non-UTF-8 output).
"""
from __future__ import annotations

import base64
import hashlib

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from app.core.exceptions import SwabiAIError


class SwabiDecryptionError(SwabiAIError):
    """Decryption failed — wrong passphrase, wrong KDF, or corrupted payload."""


_SALT_PREFIX = b"Salted__"


def decrypt_payload(
    payload_b64: str,
    passphrase: str,
    *,
    kdf: str = "md5",
) -> str:
    """Return the UTF-8 plaintext of an encrypted CryptoJS payload."""
    if not passphrase:
        raise SwabiDecryptionError("passphrase not configured")
    try:
        blob = base64.b64decode(payload_b64)
    except Exception as exc:
        raise SwabiDecryptionError(f"invalid base64: {exc}") from exc

    if not blob.startswith(_SALT_PREFIX):
        raise SwabiDecryptionError(
            f"expected 'Salted__' prefix, got {blob[:8]!r}"
        )
    salt = blob[8:16]
    ciphertext = blob[16:]

    if kdf == "md5":
        hasher = hashlib.md5
    elif kdf == "sha256":
        hasher = hashlib.sha256
    else:
        raise SwabiDecryptionError(f"unsupported kdf: {kdf}")

    password = passphrase.encode("utf-8")
    derived = b""
    prev = b""
    while len(derived) < 48:
        prev = hasher(prev + password + salt).digest()
        derived += prev
    key = derived[:32]
    iv = derived[32:48]

    try:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    except Exception as exc:
        raise SwabiDecryptionError(f"AES decryption failed: {exc}") from exc

    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SwabiDecryptionError(f"decrypted bytes are not UTF-8: {exc}") from exc
