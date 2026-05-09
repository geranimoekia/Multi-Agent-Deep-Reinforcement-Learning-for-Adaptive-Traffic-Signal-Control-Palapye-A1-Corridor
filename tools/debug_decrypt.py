"""Debug cookie decryption to find the right key/format for Vivaldi."""
import base64, ctypes, json, shutil, sqlite3, sys
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

VIVALDI_DIR  = Path.home() / "AppData/Local/Vivaldi/User Data/Default"
COOKIES_PATH = VIVALDI_DIR / "Network" / "Cookies"
LOCAL_STATE  = VIVALDI_DIR.parent / "Local State"

class DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.c_char_p)]

def dpapi_decrypt(data):
    blob_in  = DATA_BLOB(len(data), data)
    blob_out = DATA_BLOB()
    ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
    return ctypes.string_at(blob_out.pbData, blob_out.cbData)

# Load key
state = json.loads(LOCAL_STATE.read_text(encoding="utf-8"))
enc_key_b64 = state["os_crypt"]["encrypted_key"]
enc_key = base64.b64decode(enc_key_b64)
print(f"Encrypted key length: {len(enc_key)}, prefix: {enc_key[:5]}")
aes_key = dpapi_decrypt(enc_key[5:])
print(f"AES key length: {len(aes_key)}  hex: {aes_key.hex()[:32]}...")

# Load one cookie and try to decrypt it
tmp = Path("cookie_debug_tmp.db")
shutil.copy2(COOKIES_PATH, tmp)
conn = sqlite3.connect(tmp)
cur  = conn.cursor()
cur.execute("SELECT host_key, name, encrypted_value FROM cookies WHERE host_key LIKE '%openai.com%' LIMIT 5")
rows = cur.fetchall()
conn.close()
tmp.unlink(missing_ok=True)

for host, name, enc_val in rows:
    print(f"\n--- {host} / {name} ---")
    print(f"  enc_val length: {len(enc_val)}, prefix bytes: {enc_val[:6].hex()}")
    prefix = enc_val[:3]
    if prefix in (b"v10", b"v11"):
        nonce12 = enc_val[3:15]
        cipher12 = enc_val[15:]
        nonce16 = enc_val[3:19]
        cipher16 = enc_val[19:]
        # Try 12-byte nonce
        try:
            result = AESGCM(aes_key).decrypt(nonce12, cipher12, None)
            print(f"  v10 nonce12 OK: {result[:60]}")
        except InvalidTag:
            print(f"  v10 nonce12: InvalidTag")
        except Exception as e:
            print(f"  v10 nonce12 error: {e}")
        # Try 16-byte nonce
        try:
            result = AESGCM(aes_key).decrypt(nonce16, cipher16, None)
            print(f"  v10 nonce16 OK: {result[:60]}")
        except InvalidTag:
            print(f"  v10 nonce16: InvalidTag")
        except Exception as e:
            print(f"  v10 nonce16 error: {e}")
    # Also try raw DPAPI
    try:
        result = dpapi_decrypt(enc_val)
        print(f"  DPAPI raw OK: {result[:60]}")
    except Exception as e:
        print(f"  DPAPI raw: {e}")
