#!/usr/bin/env python3
"""
Author: Porygon-Tech https://github.com/porygon-tech
Vectorized extraction and embedding scripts for LSB steganography with complexity threshold.
"""

# === extraction_vectorized.py ===
import numpy as np
from PIL import Image
import sys
import os

# Parameters
BS = 8
EMBED_MIN_PLANE = 1
EMBED_MAX_PLANE = 5
THRESHOLD = 34
# Checkerboard patterns: 2D and flattened
checkerboard2d = (np.indices((BS, BS)).sum(axis=0) & 1).astype(np.uint8)
checkerboard_flat = checkerboard2d.flatten()


def bit_plane_blocks(channel, plane):
    """
    Extract bit-plane and reshape into blocks of shape (hb, wb, BS, BS),
    cropping any extra pixels if dimensions aren’t multiples of BS.
    Returns blocks axis-ordered as (hb, wb, BS, BS).
    """
    bits = ((channel >> plane) & 1).astype(np.uint8)
    h, w = bits.shape
    hb, wb = h // BS, w // BS
    bits_c = bits[:hb * BS, :wb * BS]
    return bits_c.reshape(hb, BS, wb, BS).transpose(0, 2, 1, 3)


def compute_complexity_blocks(blocks):
    """Compute complexity per block via horizontal+vertical transitions."""
    dh = np.abs(blocks[:, :, :, :-1] - blocks[:, :, :, 1:]).sum(axis=(2, 3))
    dv = np.abs(blocks[:, :, :-1, :] - blocks[:, :, 1:, :]).sum(axis=(2, 3))
    return dh + dv

# --- extraction_vectorized.py ---
def extract_secret(stego_path):
    import os
    import numpy as np
    from PIL import Image

    def bits_to_int(bits):
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return v

    img = Image.open(stego_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)

    secret_bits = []
    done = False
    have_total_need = False
    total_need = None

    for ch in range(3):
        if done: break
        channel = arr[:, :, ch]
        for plane in range(EMBED_MIN_PLANE, EMBED_MAX_PLANE + 1):
            if done: break
            blocks = bit_plane_blocks(channel, plane)
            comp = compute_complexity_blocks(blocks)
            hb, wb = comp.shape

            for i in range(hb):
                if done: break
                for j in range(wb):
                    if comp[i, j] < THRESHOLD:
                        continue

                    base_y, base_x = i * BS, j * BS
                    vals = []
                    for k in range(BS * BS):
                        r, c = divmod(k, BS)
                        v = arr[base_y + r, base_x + c, ch]
                        vals.append((v >> plane) & 1)

                    flag = arr[base_y, base_x, ch] & 1
                    sec = np.array(vals[1:], dtype=np.uint8)
                    if flag == 1:
                        sec ^= checkerboard_flat[1:]

                    secret_bits.extend(sec.tolist())

                    if not have_total_need and len(secret_bits) >= 8:
                        name_len = bits_to_int(secret_bits[:8])
                        need_header = 8 + name_len * 8 + 32
                        if len(secret_bits) >= need_header:
                            idx = 8 + name_len * 8
                            size_bits = secret_bits[idx:idx + 32]
                            payload_size = bits_to_int(size_bits)
                            total_need = need_header + payload_size * 8
                            have_total_need = True

                    if have_total_need and len(secret_bits) >= total_need:
                        done = True
                        break

    idx = 0
    name_len = bits_to_int(secret_bits[idx:idx + 8]); idx += 8
    name_bytes = bytearray()
    for _ in range(name_len):
        b = bits_to_int(secret_bits[idx:idx + 8]); idx += 8
        name_bytes.append(b)
    embedded_name = name_bytes.decode('utf-8', 'replace') or 'recovered.bin'
    size = bits_to_int(secret_bits[idx:idx + 32]); idx += 32

    data = bytearray()
    for _ in range(size):
        b = bits_to_int(secret_bits[idx:idx + 8]); idx += 8
        data.append(b)

    with open(embedded_name, 'wb') as f:
        f.write(data)
    print(f'Extracted {size} bytes to {embedded_name}')


def main():
    import sys
    if len(sys.argv) != 2:
        print(f'Usage: Extract: {sys.argv[0]} stego.png')
        sys.exit(1)
    extract_secret(sys.argv[1])

if __name__ == '__main__':
    main()