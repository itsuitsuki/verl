"""Decide WHERE the straggler lag is: measure each judge call's round-trip
(request -> response complete) and the idle gaps between calls, on loopback,
without ptrace (AF_PACKET / CAP_NET_RAW).

If per-call latency is high (~25s) -> the JUDGE (LLM generation) is the bottleneck.
If per-call latency is low (~2s) but gaps are long -> the lag is BETWEEN calls
(Isabelle prover / waits), NOT the judge.

Usage: python judge_latency.py [duration_s]
"""
import socket
import struct
import sys
import time

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
PORTS = {4873, 4874}

s = socket.socket(socket.AF_PACKET, socket.SOCK_DGRAM, socket.ntohs(0x0003))
s.settimeout(1.0)
t0 = time.time()
events = []   # (t, kind, port)  kind in {REQ, RSP}

print(f"[lat] measuring judge round-trips on ports {sorted(PORTS)} for {DUR}s ...", flush=True)
while time.time() - t0 < DUR:
    try:
        data, addr = s.recvfrom(65535)
    except socket.timeout:
        continue
    now = time.time() - t0
    if addr[0] != "lo" or len(data) < 20 or (data[0] >> 4) != 4 or data[9] != 6:
        continue
    ihl = (data[0] & 0xf) * 4
    tcp = data[ihl:]
    if len(tcp) < 20:
        continue
    sport, dport = struct.unpack("!HH", tcp[0:4])
    doff = (tcp[12] >> 4) * 4
    payload = tcp[doff:]
    if not payload:
        continue
    if dport in PORTS and b"POST" in payload[:16]:
        events.append((now, "REQ", dport))
    elif sport in PORTS:
        events.append((now, "RSP", sport))

# ---- reduce to per-call round-trips ----
# A judge call = a REQ, then a burst of RSP packets; the call completes at the
# last RSP before the next REQ / a >3s silence.
events.sort()
calls = []          # (t_req, t_resp_done, port)
i = 0
reqs = [e for e in events if e[1] == "REQ"]
rsps = [e for e in events if e[1] == "RSP"]
for k, (treq, _, port) in enumerate(reqs):
    tnext = reqs[k + 1][0] if k + 1 < len(reqs) else DUR + 1
    burst = [t for (t, kind, p) in rsps if p == port and treq <= t < tnext]
    tdone = max(burst) if burst else None
    calls.append((treq, tdone, port))

print(f"\n[lat] captured {len(reqs)} judge requests, {len(rsps)} response packets\n", flush=True)
prev_done = None
for k, (treq, tdone, port) in enumerate(calls):
    lat = (tdone - treq) if tdone is not None else None
    gap = (treq - prev_done) if prev_done is not None else None
    lat_s = f"{lat:6.1f}s" if lat is not None else "  (no resp in window)"
    gap_s = f"{gap:6.1f}s idle before this call" if gap is not None else ""
    print(f"  call#{k+1:2d} :{port}  t_req={treq:6.1f}s  round_trip={lat_s}   {gap_s}", flush=True)
    if tdone is not None:
        prev_done = tdone

lats = [t2 - t1 for (t1, t2, _) in calls if t2 is not None]
gaps = []
pd = None
for (t1, t2, _) in calls:
    if pd is not None:
        gaps.append(t1 - pd)
    if t2 is not None:
        pd = t2
if lats:
    print(f"\n[lat] round-trip: n={len(lats)} min={min(lats):.1f}s "
          f"max={max(lats):.1f}s mean={sum(lats)/len(lats):.1f}s", flush=True)
if gaps:
    print(f"[lat] idle gaps between calls: n={len(gaps)} min={min(gaps):.1f}s "
          f"max={max(gaps):.1f}s mean={sum(gaps)/len(gaps):.1f}s", flush=True)
print("\n[lat] verdict: high round_trip -> JUDGE is the lag; "
      "low round_trip + big gaps -> lag is BETWEEN calls (prover/wait).", flush=True)
