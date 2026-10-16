"""P0 probe 2: every route off the container must fail; loopback is the working control."""
import json
import os
import socket

TIMEOUT = 3.0
socket.setdefaulttimeout(TIMEOUT)


def attempt(function):
    try:
        function()
    except Exception as error:  # the probe records any failure type
        return {"connected": False, "error": "%s:%s" % (type(error).__name__, getattr(error, "errno", None))}
    return {"connected": True, "error": None}


def tcp(address):
    def connect():
        with socket.create_connection(address, timeout=TIMEOUT):
            pass
    return connect


def udp_roundtrip(address):
    def exchange():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(TIMEOUT)
            sock.sendto(b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01", address)
            sock.recvfrom(512)
    return exchange


def loopback():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    with socket.create_connection(server.getsockname(), timeout=TIMEOUT):
        pass
    server.close()


default_routes = []
for line in open("/proc/net/route").read().splitlines()[1:]:
    fields = line.split()
    if fields[1] == "00000000":
        default_routes.append(fields[0])

# Podman writes host.containers.internal into /etc/hosts even without a network, so the name
# resolving is not egress. What must fail is every connection to the addresses it names.
try:
    gateway_addresses = sorted({a[4][0] for a in socket.getaddrinfo("host.containers.internal", 80)})
except socket.gaierror:
    gateway_addresses = []

result = {
    "dns_public_name": attempt(lambda: socket.getaddrinfo("pypi.org", 443)),
    "host_gateway_name_resolves_to": gateway_addresses,
    "tcp_host_gateway": {"%s:%d" % (a, port): attempt(tcp((a, port)))
                         for a in gateway_addresses for port in (22, 53, 80, 443)},
    "dns_udp_to_public_resolver": attempt(udp_roundtrip(("8.8.8.8", 53))),
    "tcp_public_ip_1": attempt(tcp(("1.1.1.1", 443))),
    "tcp_public_ip_2": attempt(tcp(("8.8.8.8", 53))),
    "tcp_cloud_metadata": attempt(tcp(("169.254.169.254", 80))),
    "tcp_podman_default_gateway": attempt(tcp(("10.88.0.1", 80))),
    "tcp_podman_host_alias_ip": attempt(tcp(("192.168.127.1", 80))),
    "loopback_control": attempt(loopback),
    "interfaces": sorted(os.listdir("/sys/class/net")),
    "default_routes": default_routes,
}
json.dump(result, open("/out/result.json", "w"), sort_keys=True)
