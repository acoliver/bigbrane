"""P0 probe 5c: outlive the wall cap."""
import time

print("sleeping", flush=True)
time.sleep(120)
print("woke (should never print)", flush=True)
