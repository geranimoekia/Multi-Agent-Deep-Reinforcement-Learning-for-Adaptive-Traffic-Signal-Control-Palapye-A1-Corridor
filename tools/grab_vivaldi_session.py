"""
grab_vivaldi_session.py
Extract Prism/OpenAI cookies from Vivaldi and save as prism_session.json.

IMPORTANT: Close Vivaldi before running, or it may lock the cookie database.

Usage:
    rl_env\Scripts\python.exe tools\grab_vivaldi_session.py
"""
import base64
import ctypes
import json
import shutil
import sqlite3
import struct
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
VIVALDI_DIR  = Path.home() / "AppData/Local/Vivaldi/User Data/Default"
COOKIES_PATH = VIVALDI_DIR / "Network" / "Cookies"
LOCAL_STATE  = VIVALDI_DIR.parent / "Local State"
SESSION_FILE = Path("prism_session.json")

TARGET_DOMAINS = ["prism.openai.com", "openai.com", ".openai.com", "auth.openai.com"]

# ── Windows DPAPI decrypt ─────────────────────────────────────────────────────

def dpapi_decrypt(ciphertext: bytes) -> bytes:
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.c_char_p)]

    blob_in  = DATA_BLOB(len(ciphertext), ciphertext)
    blob_out = DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0,
        ctypes.byref(blob_out)
    )
    if not ok:
        raise RuntimeError("DPAPI decryption failed")
    return ctypes.string_at(blob_out.pbData, blob_out.cbData)


def get_aes_key() -> bytes:
    state = json.loads(LOCAL_STATE.read_text(encoding="utf-8"))
    encrypted_key_b64 = state["os_crypt"]["encrypted_key"]
    encrypted_key = base64.b64decode(encrypted_key_b64)
    # Strip "DPAPI" prefix (first 5 bytes)
    return dpapi_decrypt(encrypted_key[5:])


def decrypt_cookie(aes_key: bytes, encrypted_value: bytes) -> str:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        sys.exit("[ERROR] Run: rl_env\\Scripts\\pip.exe install cryptography")

    if encrypted_value[:3] == b"v10" or encrypted_value[:3] == b"v11":
        nonce  = encrypted_value[3:3+12]
        cipher = encrypted_value[3+12:]
        return AESGCM(aes_key).decrypt(nonce, cipher, None).decode("utf-8")
    # Older DPAPI-encrypted value
    return dpapi_decrypt(encrypted_value).decode("utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not COOKIES_PATH.exists():
        sys.exit(f"[ERROR] Cookies not found at {COOKIES_PATH}")

    print("[INFO] Reading AES key from Local State...")
    try:
        aes_key = get_aes_key()
    except Exception as e:
        sys.exit(f"[ERROR] Could not get AES key: {e}")

    # Copy the database so we don't conflict with a running Vivaldi
    tmp = Path("vivaldi_cookies_tmp.db")
    try:
        shutil.copy2(COOKIES_PATH, tmp)
    except PermissionError:
        sys.exit("[ERROR] Cannot copy Cookies file — close Vivaldi first.")

    print("[INFO] Extracting cookies for OpenAI/Prism domains...")
    cookies = []
    try:
        conn = sqlite3.connect(tmp)
        cur  = conn.cursor()
        cur.execute(
            "SELECT host_key, name, encrypted_value, path, expires_utc, "
            "       is_httponly, is_secure, samesite "
            "FROM cookies WHERE host_key LIKE '%openai.com%'"
        )
        for host, name, enc_val, path, expires, httponly, secure, samesite in cur.fetchall():
            try:
                value = decrypt_cookie(aes_key, enc_val)
            except Exception:
                value = ""
            samesite_map = {-1: "None", 0: "None", 1: "Lax", 2: "Strict"}
            cookies.append({
                "name":     name,
                "value":    value,
                "domain":   host,
                "path":     path,
                "expires":  (expires / 1_000_000 - 11644473600) if expires else -1,
                "httpOnly": bool(httponly),
                "secure":   bool(secure),
                "sameSite": samesite_map.get(samesite, "None"),
            })
        conn.close()
    finally:
        tmp.unlink(missing_ok=True)

    if not cookies:
        sys.exit("[ERROR] No openai.com cookies found. Are you logged in to Prism in Vivaldi?")

    storage_state = {"cookies": cookies, "origins": []}
    SESSION_FILE.write_text(json.dumps(storage_state, indent=2))
    print(f"[OK] Saved {len(cookies)} cookies to {SESSION_FILE}")
    print("[OK] You can now run: rl_env\\Scripts\\python.exe tools\\push_once.py")


if __name__ == "__main__":
    main()
