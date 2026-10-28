"""Round-trip tests for the CryptoJS AES decryptor."""
from __future__ import annotations

import base64
import hashlib
import os

import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from app.swabi.crypto import SwabiDecryptionError, decrypt_payload


def _encrypt(plaintext: str, passphrase: str, *, kdf: str = "md5") -> str:
    """Produce the CryptoJS envelope the frontend produces."""
    salt = os.urandom(8)
    password = passphrase.encode("utf-8")
    hasher = hashlib.md5 if kdf == "md5" else hashlib.sha256
    derived = b""
    prev = b""
    while len(derived) < 48:
        prev = hasher(prev + password + salt).digest()
        derived += prev
    key, iv = derived[:32], derived[32:48]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(plaintext.encode("utf-8"), AES.block_size))
    return base64.b64encode(b"Salted__" + salt + ct).decode("ascii")


@pytest.mark.parametrize("kdf", ["md5", "sha256"])
def test_round_trip(kdf):
    payload = '{"success": true, "data": {"escrowBalance": 0}}'
    encrypted = _encrypt(payload, "secret-pass", kdf=kdf)
    assert decrypt_payload(encrypted, "secret-pass", kdf=kdf) == payload


def test_wrong_passphrase_raises():
    encrypted = _encrypt('{"a": 1}', "correct-pass")
    with pytest.raises(SwabiDecryptionError):
        decrypt_payload(encrypted, "wrong-pass")


def test_missing_salt_prefix_raises():
    blob = base64.b64encode(b"not-salted-here").decode()
    with pytest.raises(SwabiDecryptionError, match="Salted__"):
        decrypt_payload(blob, "x")


def test_empty_passphrase_raises():
    with pytest.raises(SwabiDecryptionError, match="not configured"):
        decrypt_payload("AAAA", "")
