"""P0 probe 5b: allocate and touch memory until the kernel kills the process."""
import sys

blocks = []
while True:
    blocks.append(bytearray(b"\1") * (16 << 20))
    print("held_mb", 16 * len(blocks), flush=True)
