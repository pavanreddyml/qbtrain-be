"""
Crypto utilities for the qbtrain exfil training module.

Provides Base64-URL encoding/decoding and HS256 JWT helpers used by the
exfiltration-lab sandbox.  All functions operate on bounded inputs to
prevent denial-of-service via oversized payloads.
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Optional

# --- safety limits --------------------------------------------------------
MAX_PAYLOAD_BYTES = 64 * 1024  # 64 KiB – reject anything larger


class CryptoError(Exception):
    """Raised for any encode / decode / verification failure."""


# --- Base64-URL helpers ---------------------------------------------------

def b64url_encode(data: bytes) -> str:
    """Return unpadded Base64-URL encoding of *data*."""
    if len(data) > MAX_PAYLOAD_BYTES:
        raise CryptoError(
            f"Payload too large ({len(data)} bytes, max {MAX_PAYLOAD_BYTES})"
        )
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(s: str) -> bytes:
    """Decode a Base64-URL string (padded or unpadded)."""
    if len(s) > MAX_PAYLOAD_BYTES:
        raise CryptoError(
            f"Input too large ({len(s)} chars, max {MAX_PAYLOAD_BYTES})"
        )
    s = s.replace("-", "+").replace("_", "/")
    padding = "=" * (-len(s) % 4)
    return base64.b64decode(s + padding)


# --- HMAC-SHA256 ----------------------------------------------------------

def hmac_sha256_sign(key: str, message: bytes) -> bytes:
    """Return raw HMAC-SHA256 signature."""
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).digest()


def hmac_sha256_verify(key: str, message: bytes, signature: bytes) -> bool:
    """Constant-time verification of an HMAC-SHA256 signature."""
    expected = hmac_sha256_sign(key, message)
    return hmac.compare_digest(expected, signature)


# --- JWT (HS256 only) -----------------------------------------------------

def create_jwt(payload: dict, secret: str, ttl_seconds: int = 3600) -> str:
    """
    Create an HS256 JWT with an ``exp`` claim.

    Parameters
    ----------
    payload : dict
        Claims to include (must be JSON-serialisable).
    secret : str
        HMAC signing key.
    ttl_seconds : int
        Token lifetime in seconds (default 1 h).

    Returns
    -------
    str
        Compact JWS (``header.payload.signature``).
    """
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {**payload, "exp": int(time.time()) + ttl_seconds}

    header_b64 = b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = b64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = hmac_sha256_sign(secret, signing_input)
    sig_b64 = b64url_encode(sig)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_jwt(token: str, secret: str, *, check_exp: bool = True) -> Optional[dict]:
    """
    Verify an HS256 JWT and return its payload, or ``None`` on failure.

    Parameters
    ----------
    token : str
        Compact JWS string.
    secret : str
        HMAC signing key.
    check_exp : bool
        If *True* (default), reject expired tokens.

    Returns
    -------
    dict or None
        The decoded payload claims, or *None* if verification fails.
    """
    if len(token) > MAX_PAYLOAD_BYTES:
        return None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = b64url_decode(sig_b64)

        if not hmac_sha256_verify(secret, signing_input, sig):
            return None

        payload = json.loads(b64url_decode(payload_b64).decode("utf-8"))

        if check_exp:
            exp = payload.get("exp")
            if exp is not None and time.time() > exp:
                return None

        return payload
    except Exception:
        return None
