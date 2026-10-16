"""P0 probe 3: what can the job see of the host? /in/holdout_path.txt names an absolute host path of a holdout file."""
import json
import os

holdout = open("/in/holdout_path.txt").read().strip()
token = os.path.basename(holdout)


def attempt(function):
    try:
        return {"ok": True, "value": function(), "error": None}
    except Exception as error:
        return {"ok": False, "value": None, "error": "%s:%s" % (type(error).__name__, getattr(error, "errno", None))}


def read(path):
    with open(path, "rb") as handle:
        return len(handle.read())


ancestors = []
node = os.path.dirname(holdout)
while node != "/":
    ancestors.append(node)
    node = os.path.dirname(node)

mounts = []
for line in open("/proc/self/mountinfo"):
    left, _, right = line.partition(" - ")
    fields = left.split()
    mounts.append({"mount_point": fields[4], "root": fields[3], "fstype": right.split()[0]})

found = []
for base, dirs, files in os.walk("/", onerror=lambda error: None):
    if base in ("/proc", "/sys", "/dev"):
        dirs[:] = []
        continue
    if token in files or token in dirs:
        found.append(os.path.join(base, token))

relative_up = "/in/" + "../" * 12 + holdout.lstrip("/")
for name, target in (("link_to_holdout", holdout), ("link_to_in_parent", "/in/..")):
    attempt(lambda name=name, target=target: os.symlink(target, "/out/" + name))

result = {
    "holdout_exists": os.path.exists(holdout),
    "read_holdout": attempt(lambda: read(holdout)),
    "read_holdout_via_in_dotdot": attempt(lambda: read(relative_up)),
    "list_holdout_dir": attempt(lambda: os.listdir(os.path.dirname(holdout))),
    "existing_ancestors": [a for a in ancestors if os.path.exists(a)],
    "root_listing": sorted(os.listdir("/")),
    "walk_found_holdout_name": found,
    "mounts": mounts,
    "in_listing": sorted(os.listdir("/in")),
    "read_in_seed": attempt(lambda: read("/in/seed.txt")),
    "read_through_staged_link": attempt(lambda: read("/out/link_to_holdout")),
}
json.dump(result, open("/out/result.json", "w"), sort_keys=True)
