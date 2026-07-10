"""Loopback judge-request sniffer (no ptrace; uses CAP_NET_RAW via AF_PACKET).
Captures HTTP POST bodies going to the judge ports on `lo`, reassembles per
source port, and extracts the translation prompt so we can see WHICH problem /
steps the straggler is currently translating. Bounded by duration or N requests.

Usage: python judge_sniff.py [duration_s] [max_requests]
"""
import json
import re
import socket
import struct
import sys
import time

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
MAXREQ = int(sys.argv[2]) if len(sys.argv) > 2 else 6
PORTS = {4873, 4874}

s = socket.socket(socket.AF_PACKET, socket.SOCK_DGRAM, socket.ntohs(0x0003))
s.settimeout(3.0)
t0 = time.time()
conns = {}   # sport -> bytearray

print(f"[sniff] capturing dst ports {sorted(PORTS)} on lo for up to {DUR}s ...", flush=True)
while time.time() - t0 < DUR:
    try:
        data, addr = s.recvfrom(65535)
    except socket.timeout:
        continue
    if addr[0] != "lo" or len(data) < 20:
        continue
    if (data[0] >> 4) != 4 or data[9] != 6:   # IPv4 + TCP
        continue
    ihl = (data[0] & 0xf) * 4
    tcp = data[ihl:]
    if len(tcp) < 20:
        continue
    sport, dport = struct.unpack("!HH", tcp[0:4])
    doff = (tcp[12] >> 4) * 4
    payload = tcp[doff:]
    if dport in PORTS and payload:
        conns.setdefault(sport, bytearray()).extend(payload)

def json_strings_for_key(body, key='"content":'):
    """Escape-aware extraction of every JSON string value following `key`."""
    out, i = [], 0
    while True:
        j = body.find(key, i)
        if j < 0:
            break
        k = body.find('"', j + len(key))
        if k < 0:
            break
        m, buf = k + 1, []
        while m < len(body):
            c = body[m]
            if c == "\\" and m + 1 < len(body):
                buf.append(body[m:m + 2]); m += 2; continue
            if c == '"':
                break
            buf.append(c); m += 1
        out.append("".join(buf))
        i = m + 1
    return out


def unescape(s):
    return (s.replace("\\n", "\n").replace("\\t", "\t")
             .replace('\\"', '"').replace("\\\\", "\\"))


# ---- extract translation prompts from captured request bodies ----
printed = 0
for sport, buf in conns.items():
    text = bytes(buf).decode("latin1", "replace")
    for chunk in re.split(r"(?=POST /v1/chat/completions)", text):
        if "chat/completions" not in chunk:
            continue
        body = chunk.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in chunk else chunk
        contents = json_strings_for_key(body)
        if not contents:
            continue
        # the USER message (with the problem/steps) is the longest content field
        prompt = unescape(max(contents, key=len))
        kind = "STEPS" if ("STEP TRANSCRIPTION" in prompt or "NEW STEPS" in prompt) \
            else ("GIVENS" if "GIVEN" in prompt else "?")
        printed += 1
        print(f"\n===== request #{printed} sport={sport} kind={kind} "
              f"content_len={len(prompt)} (fields={[len(c) for c in contents]}) =====",
              flush=True)
        # print head (instructions) truncated + FULL tail (where the actual
        # problem / steps live)
        if len(prompt) > 2600:
            print(prompt[:600] + "\n  ...[snip]...\n" + prompt[-2000:], flush=True)
        else:
            print(prompt, flush=True)
        if printed >= MAXREQ:
            break
    if printed >= MAXREQ:
        break

print(f"\n[sniff] done. connections={len(conns)} requests_extracted={printed}", flush=True)
