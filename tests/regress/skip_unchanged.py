"""Checks SKIP_UNCHANGED_CHUNKS on one image, on fresh servers and data, and
prints every problem.

    python skip_unchanged.py --image IMAGE --work DIR [--platform linux/amd64]
                             [--rounds 40] [--require-hook]

With SKIP_UNCHANGED_CHUNKS=1 a PATCH reads the chunks it covers whole as well,
and leaves one it does not change as stored, after checking under the chunk
cache lock that its .meta entry is still the one its voxels were read under.
Each case below runs on a server with the switch on and, where it says so, on
one with it off (the control). Every layer is one mchunk of 128x128x64 voxels,
4x4x2 chunks of 32^3.

- a: a PATCH of the whole layer with exactly what it holds appends nothing to
  .data and leaves .meta as it was (on); the same PATCH appends 32 frames
  (off). Also on a layer never written, with zeros.
- b: the same with one voxel changed: exactly one frame is appended, exactly
  that chunk's entry moves (to the old end of .data), and the layer reads
  back as sent.
- c: a box that covers two chunks whole and others in part, one of the
  partly covered ones changed: the layer reads back with the box merged in,
  the same on and off.
- d1, d1b, d2 (only on a build made with -DNTRACER_TEST_HOOKS=ON; a PATCH
  then waits before its skip check while /data/.test_hold_skip_check exists):
  d1, a PATCH A that sends a chunk's own voxels is held there while a PATCH B
  writes other voxels into that chunk; A must still land, as it does without
  the switch, where A's write always comes after B's. d1b, the same on a
  chunk never written, with zeros. d2, while A is held the mchunk's .meta and
  .data are replaced by copies (new files with the same bytes); A must answer
  500 "Could not write chunk", as a write into replaced files does.
- e1: zeros into an mchunk whose .data is missing answer 500, on and off,
  and the next write elsewhere still goes through. e2: zeros over a chunk
  whose zstd frame is corrupt replace it: it reads back as zeros and a
  PATCH of part of it is then merged. e2b: a whole-layer PATCH of a layer
  with one corrupt chunk and its own voxels everywhere (read from an intact
  copy) rewrites only that chunk and leaves the other 31 as stored. e4: an mchunk whose header names video
  compression is written as without the switch (its frames do not decode as
  video either, so this cannot tell "not read" from "read failed"). e5 (hook
  build): while a skip check waits inside the chunk cache lock
  (/data/.test_hold_write_lock), /health answers 503 with the time the lock
  has been held, not -1, and the PATCH then leaves the chunk as stored.
- s19h: harness s19 (8 PATCHes at once into one mchunk, 40 rounds) with half
  the clients sending the same contents again in every round after the
  first: no problem, and the counters show every resent chunk left as
  stored while the other half were written beside it.
- counters and settings: /performance shows "Chunks covered whole by writes:
  N; left as stored because unchanged: M" only with the switch on, and the
  counts move by what each case covered and skipped; off, neither that line
  nor any log line of the switch appears; "0" is off without a log line;
  any other value is off with one "SKIP_UNCHANGED_CHUNKS ignored" line. In
  a, b, c, e2b and s19h the counts must move by exactly what the case covered
  and skipped.

A case that raises is recorded as a problem and the next one runs. At the
end every saved server log is searched for AddressSanitizer reports (an
ASan build), and the UBSan "runtime error:" sites are listed.

Exit status 1 if any case found a problem. Where a PATCH never stops at the
hook the d and e5 cases are reported as SKIPPED (a build without the hook,
or, for d, a PATCH with nothing to skip, which case a would also show), and
--require-hook makes that a failure.
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np

import fixtures
import harness

SIZE = (128, 128, 64)
C = 32
GRID = (4, 4, 2)
NAME = "chunk_0_0_0.0.1X"
WHOLE = harness.box(0, SIZE[0], 0, SIZE[1], 0, SIZE[2])
SKIP_MARKER = "/data/.test_hold_skip_check"
LOCK_MARKER = "/data/.test_hold_write_lock"
SKIP_HOOK_LINE = "TEST HOOK: write held before its skip check"
LOCK_HOOK_LINE = "TEST HOOK: write holding the chunk cache lock"
PERF = re.compile(r"<p>Chunks covered whole by writes: (\d+); left as stored because unchanged: (\d+)</p>")
STUCK = re.compile(r"stuck lock_wait_ms=(\d+) write_held_ms=(-?\d+)\n")
REPLACED = b"500 Internal Server Error -- Could not write chunk\n"


def wpath(ds, b):
    return f"/{ds}+token={fixtures.TOKEN}/write/1/{b}"


def rpath(ds, b):
    return f"/{ds}+token={fixtures.TOKEN}/1/{b}"


def chunk_box(cx, cy, cz):
    return harness.box(cx * C, cx * C + C, cy * C, cy * C + C, cz * C, cz * C + C)


def chunk_id(cx, cy, cz):
    return (cx * GRID[1] + cy) * GRID[2] + cz


def read(server, ds, b, shape):
    st, body = server.request("GET", rpath(ds, b))
    if st != 200:
        return st, None
    return st, np.frombuffer(body, dtype=np.uint16).reshape(shape).copy()  # [z, y, x]


def files_of(server, ds):
    """(.data bytes, .meta bytes) as the server's container sees them."""
    cat = lambda ext: subprocess.run(["docker", "exec", server.name, "cat", f"/data/{ds}/{ext}/{NAME}.{ext}"],
                                     check=True, capture_output=True).stdout
    return cat("data"), cat("meta")


def entries(meta):
    header = struct.calcsize(fixtures.SHARD_HEADER_LAYOUT)
    line = struct.calcsize(fixtures.SHARD_LINE_LAYOUT)
    return [struct.unpack(fixtures.SHARD_LINE_LAYOUT, meta[header + line * i:header + line * (i + 1)])
            for i in range((len(meta) - header) // line)]


def log_lines(server):
    r = subprocess.run(["docker", "logs", server.name], capture_output=True, text=True)
    return (r.stdout + r.stderr).splitlines()


def perf_counts(server):
    st, body = server.request("GET", "/performance")
    m = PERF.search(body.decode("utf-8", "replace")) if st == 200 else None
    return (int(m.group(1)), int(m.group(2))) if m else None


class Held:
    """Sends a PATCH on its own thread while marker exists, and waits until the
    server logs hook_line for it. held is False on a build without the hook
    (the PATCH then answers at once)."""

    def __init__(self, server, marker, hook_line, path, body):
        self.server, self.marker = server, marker
        before = sum(hook_line in x for x in log_lines(server))
        server.exec("touch", marker)
        self.result = []
        self.thread = threading.Thread(target=lambda: self.result.append(server.request("PATCH", path, body)))
        self.thread.start()
        self.held = False
        deadline = time.monotonic() + 10
        while not self.held and self.thread.is_alive() and time.monotonic() < deadline:
            self.held = sum(hook_line in x for x in log_lines(server)) > before
            time.sleep(0.1)

    def release(self):
        self.server.exec("rm", "-f", self.marker)
        self.thread.join(130)
        return self.result[0] if self.result else ("NO ANSWER", b"")


def counted(on, P, tag, covered, skipped):
    """Returns a function that checks the counters moved by covered and skipped."""
    before = perf_counts(on)

    def check():
        after = perf_counts(on)
        if before is None or after is None:
            P.append(f"{tag}: /performance has no counter line (before {before}, after {after})")
            return
        moved = (after[0] - before[0], after[1] - before[1])
        print(f"  {tag}: counters moved by {moved} (covered, skipped)")
        if moved != (covered, skipped):
            P.append(f"{tag}: counters moved by {moved} (covered, skipped), want {(covered, skipped)}")

    return check


def case_a(on, off, P):
    for ds, zeros in (("a_pre", False), ("a_empty", True)):
        for server in (on, off):
            tag = f"a {ds} ({'on' if server is on else 'off'})"
            st, vol = read(server, ds, WHOLE, SIZE[::-1])
            if st != 200:
                P.append(f"{tag}: read answered {st}")
                continue
            if zeros and vol.any():
                P.append(f"{tag}: the layer never written reads back non-zero")
            d0, m0 = files_of(server, ds)
            check = counted(on, P, tag, 32, 32) if server is on else None
            st, body = server.request("PATCH", wpath(ds, WHOLE), vol.tobytes())
            d1, m1 = files_of(server, ds)
            st2, back = read(server, ds, WHOLE, SIZE[::-1])
            print(f"  {tag}: PATCH {st} {body!r}; .data {len(d0)} -> {len(d1)} B; .meta "
                  f"{'unchanged' if m1 == m0 else 'changed'}")
            if (st, body) != (200, b"done."):
                P.append(f"{tag}: PATCH answered {st} {body!r}")
            if st2 != 200 or not np.array_equal(back, vol):
                P.append(f"{tag}: the layer reads back other than sent ({st2})")
            if server is on:
                check()
                if d1 != d0 or m1 != m0:
                    P.append(f"{tag}: .data {len(d0)} -> {len(d1)} B, .meta "
                             f"{'unchanged' if m1 == m0 else 'changed'}; want both unchanged")
            else:
                moved = sum(a != b for a, b in zip(entries(m0), entries(m1)))
                if len(d1) <= len(d0) or moved != 32:
                    P.append(f"{tag}: .data {len(d0)} -> {len(d1)} B with {moved} entries moved; want it to "
                             "grow with all 32 (the control: without the switch every chunk covered whole "
                             "is written)")


def case_b(on, P):
    ds, tag = "b_pre", "b"
    st, vol = read(on, ds, WHOLE, SIZE[::-1])
    new = vol.copy()
    new[5, 7, 40] += 1  # z 5, y 7, x 40: chunk (1, 0, 0)
    want_id = chunk_id(1, 0, 0)
    d0, m0 = files_of(on, ds)
    check = counted(on, P, tag, 32, 31)
    st, body = on.request("PATCH", wpath(ds, WHOLE), new.tobytes())
    d1, m1 = files_of(on, ds)
    e0, e1 = entries(m0), entries(m1)
    moved = [i for i, (a, b) in enumerate(zip(e0, e1)) if a != b]
    st2, back = read(on, ds, WHOLE, SIZE[::-1])
    print(f"  {tag}: PATCH {st} {body!r}; .data {len(d0)} -> {len(d1)} B; entries moved {moved}"
          f"{'; chunk ' + str(want_id) + ' now ' + str(e1[want_id]) if moved else ''}")
    if (st, body) != (200, b"done."):
        P.append(f"{tag}: PATCH answered {st} {body!r}")
    if moved != [want_id] or e1[want_id] != (len(d0), len(d1) - len(d0)) or e0[want_id] == e1[want_id]:
        P.append(f"{tag}: entries moved {moved} (chunk {want_id}: {e0[want_id]} -> {e1[want_id]}, .data "
                 f"{len(d0)} -> {len(d1)} B); want only chunk {want_id}, at the old end of .data with the "
                 "growth as its size")
    if st2 != 200 or not np.array_equal(back, new):
        P.append(f"{tag}: the layer reads back other than sent ({st2})")
    check()


def case_c(on, off, P):
    ds = "c_pre"
    x0, x1, y0, y1, z0, z1 = 10, 100, 5, 70, 0, 40  # whole: chunks (1, 1, 0) and (2, 1, 0)
    b = harness.box(x0, x1, y0, y1, z0, z1)
    out = {}
    for server in (on, off):
        tag = f"c ({'on' if server is on else 'off'})"
        st, vol0 = read(server, ds, WHOLE, SIZE[::-1])
        st, sub = read(server, ds, b, (z1 - z0, y1 - y0, x1 - x0))
        mod = sub.copy()
        mod[0:5, 0:3, 0:5] += 1  # inside chunk (0, 0, 0), which the box covers in part
        d0, _ = files_of(server, ds)
        check = counted(on, P, tag, 2, 2) if server is on else None
        st, body = server.request("PATCH", wpath(ds, b), mod.tobytes())
        d1, _ = files_of(server, ds)
        st2, vol1 = read(server, ds, WHOLE, SIZE[::-1])
        want = vol0.copy()
        want[z0:z1, y0:y1, x0:x1] = mod
        print(f"  {tag}: PATCH {st} {body!r}; .data {len(d0)} -> {len(d1)} B")
        if (st, body) != (200, b"done."):
            P.append(f"{tag}: PATCH answered {st} {body!r}")
        if st2 != 200 or not np.array_equal(vol1, want):
            P.append(f"{tag}: the layer does not read back with the box merged in ({st2})")
        if check:
            check()
        out[server is on] = vol1
    if out.get(True) is not None and out.get(False) is not None and not np.array_equal(out[True], out[False]):
        P.append("c: the layer reads back differently on and off")


def case_d1(on, P, skipped, ds, tag):
    cb = chunk_box(0, 0, 0)
    st, x = read(on, ds, cb, (C, C, C))
    y = x + 1
    held = Held(on, SKIP_MARKER, SKIP_HOOK_LINE, wpath(ds, cb), x.tobytes())
    try:
        if not held.held:
            if held.thread.is_alive():
                P.append(f"{tag}: PATCH A neither answered nor reached the hook within 10 s")
            else:
                print(f"  {tag}: SKIPPED: A answered {held.result[0][0]} without stopping at the hook (no test hook "
                      "in this build, or A had no chunk to skip)")
                skipped.append(tag)
            return
        st_b, body_b = on.request("PATCH", wpath(ds, cb), y.tobytes(), timeout=20)
        st, mid = read(on, ds, cb, (C, C, C))
        if (st_b, body_b) != (200, b"done.") or st != 200 or not np.array_equal(mid, y):
            P.append(f"{tag}: PATCH B answered {st_b} {body_b!r} and the chunk does not read as B's while A is held")
    finally:
        st_a, body_a = held.release()
    st, final = read(on, ds, cb, (C, C, C))
    holds = "A's" if st == 200 and np.array_equal(final, x) else "B's" if st == 200 and np.array_equal(final, y) else "other"
    print(f"  {tag}: A held, B {st_b} {body_b!r}, A released {st_a} {body_a!r}; the chunk holds {holds} voxels")
    if (st_a, body_a) != (200, b"done."):
        P.append(f"{tag}: PATCH A answered {st_a} {body_a!r}")
    if holds != "A's":
        P.append(f"{tag}: the chunk holds {holds} voxels; want A's, which land after B's as without the switch")


def case_d2(on, P, skipped):
    ds, tag = "d2_pre", "d2"
    cb = chunk_box(0, 0, 0)
    st, x = read(on, ds, cb, (C, C, C))
    held = Held(on, SKIP_MARKER, SKIP_HOOK_LINE, wpath(ds, cb), x.tobytes())
    try:
        if not held.held:
            if held.thread.is_alive():
                P.append(f"{tag}: PATCH A neither answered nor reached the hook within 10 s")
            else:
                print(f"  {tag}: SKIPPED: A answered {held.result[0][0]} without stopping at the hook (no test hook "
                      "in this build, or A had no chunk to skip)")
                skipped.append(tag)
            return
        d0, m0 = files_of(on, ds)
        on.exec("sh", "-c", f"cd /data/{ds} && cp data/{NAME}.data data/{NAME}.data.new && "
                            f"cp meta/{NAME}.meta meta/{NAME}.meta.new && mv data/{NAME}.data.new data/{NAME}.data && "
                            f"mv meta/{NAME}.meta.new meta/{NAME}.meta")
    finally:
        st_a, body_a = held.release()
    d1, m1 = files_of(on, ds)
    st, final = read(on, ds, cb, (C, C, C))
    print(f"  {tag}: A held, files replaced by copies, A released {st_a} {body_a!r}; files "
          f"{'as copied' if (d1, m1) == (d0, m0) else 'changed'}; chunk read {st}")
    if (st_a, body_a) != (500, REPLACED):
        P.append(f"{tag}: PATCH A answered {st_a} {body_a!r}; want 500 {REPLACED!r}, as a write into replaced files")
    if (d1, m1) != (d0, m0):
        P.append(f"{tag}: the replaced files changed after A")
    if st != 200 or not np.array_equal(final, x):
        P.append(f"{tag}: the chunk reads back {st}, not its voxels")


def case_e1(on, off, P):
    ds = "e1_nodata"
    zeros = np.zeros((C, C, C), dtype=np.uint16).tobytes()
    ones = np.ones((C, C, C), dtype=np.uint16).tobytes()
    for server in (on, off):
        tag = f"e1 ({'on' if server is on else 'off'})"
        server.exec("rm", f"/data/{ds}/data/{NAME}.data")
        st, body = server.request("PATCH", wpath(ds, chunk_box(0, 0, 0)), zeros, timeout=20)
        st2, body2 = server.request("PATCH", wpath("e1_other", chunk_box(1, 1, 1)), ones, timeout=20)
        print(f"  {tag}: zeros into the mchunk with no .data {st} {body!r}; next PATCH elsewhere {st2} {body2!r}")
        if (st, body) != (500, REPLACED):
            P.append(f"{tag}: answered {st} {body!r}; want 500 {REPLACED!r}")
        if (st2, body2) != (200, b"done."):
            P.append(f"{tag}: the next PATCH elsewhere answered {st2} {body2!r}")


def case_e2(on, P):
    ds, tag = "e2_corrupt", "e2"
    cb = chunk_box(0, 0, 0)
    d0, _ = files_of(on, ds)
    st, body = on.request("PATCH", wpath(ds, cb), np.zeros((C, C, C), dtype=np.uint16).tobytes())
    d1, m1 = files_of(on, ds)
    st2, back = read(on, ds, cb, (C, C, C))
    one = harness.box(3, 4, 4, 5, 5, 6)
    st3, body3 = on.request("PATCH", wpath(ds, one), np.array([9], dtype=np.uint16).tobytes())
    st4, back2 = read(on, ds, cb, (C, C, C))
    want = np.zeros((C, C, C), dtype=np.uint16)
    want[5, 4, 3] = 9
    print(f"  {tag}: zeros over the corrupt chunk {st} {body!r} (.data {len(d0)} -> {len(d1)} B, entry "
          f"{entries(m1)[0]}); read {st2}; one voxel into it {st3} {body3!r}; read {st4}")
    if (st, body) != (200, b"done.") or len(d1) <= len(d0) or entries(m1)[0][0] != len(d0):
        P.append(f"{tag}: the corrupt chunk was not replaced ({st} {body!r}, .data {len(d0)} -> {len(d1)} B)")
    if st2 != 200 or back is None or back.any():
        P.append(f"{tag}: the replaced chunk reads back {st2}, not zeros")
    if (st3, body3) != (200, b"done.") or st4 != 200 or not np.array_equal(back2, want):
        P.append(f"{tag}: a PATCH of part of the replaced chunk answered {st3} {body3!r} or did not merge")


def case_e2b(on, P):
    ds, tag = "e2b_corrupt", "e2b"
    st, vol = read(on, "e2b_twin", WHOLE, SIZE[::-1])  # the same voxels, intact
    d0, m0 = files_of(on, ds)
    check = counted(on, P, tag, 32, 31)
    st, body = on.request("PATCH", wpath(ds, WHOLE), vol.tobytes())
    d1, m1 = files_of(on, ds)
    e0, e1 = entries(m0), entries(m1)
    moved = [i for i, (a, b) in enumerate(zip(e0, e1)) if a != b]
    st2, back = read(on, ds, WHOLE, SIZE[::-1])
    print(f"  {tag}: whole-layer PATCH over one corrupt chunk {st} {body!r}; .data {len(d0)} -> {len(d1)} B; "
          f"entries moved {moved}; read {st2}")
    if (st, body) != (200, b"done."):
        P.append(f"{tag}: PATCH answered {st} {body!r}")
    if moved != [0] or e1[0] != (len(d0), len(d1) - len(d0)):
        P.append(f"{tag}: entries moved {moved} (.data {len(d0)} -> {len(d1)} B); want only the corrupt chunk 0, "
                 "rewritten at the old end of .data")
    if st2 != 200 or not np.array_equal(back, vol):
        P.append(f"{tag}: the layer reads back {st2}, not its voxels")
    check()


def case_e4(on, P):
    ds, tag = "e4_video", "e4"
    d0, _ = files_of(on, ds)
    st, body = on.request("PATCH", wpath(ds, chunk_box(1, 1, 1)), np.zeros((C, C, C), dtype=np.uint16).tobytes())
    d1, _ = files_of(on, ds)
    print(f"  {tag}: zeros into a chunk of an mchunk whose header names video compression {st} {body!r}; "
          f".data {len(d0)} -> {len(d1)} B")
    if (st, body) != (200, b"done.") or len(d1) <= len(d0):
        P.append(f"{tag}: answered {st} {body!r} with .data {len(d0)} -> {len(d1)} B; want it written")


def case_e5(on, P, skipped):
    ds, tag = "e5_pre", "e5"
    cb = chunk_box(0, 0, 0)
    st, x = read(on, ds, cb, (C, C, C))
    d0, m0 = files_of(on, ds)
    held = Held(on, LOCK_MARKER, LOCK_HOOK_LINE, wpath(ds, cb), x.tobytes())
    try:
        if not held.held:
            if held.thread.is_alive():
                P.append(f"{tag}: the PATCH neither answered nor reached the hook within 10 s")
            else:
                print(f"  {tag}: SKIPPED: the PATCH answered {held.result[0][0]} without stopping at the hook "
                      "(no test hook in this build)")
                skipped.append(tag)
            return
        t0 = time.monotonic()
        st_h, body_h = on.request("GET", "/health", timeout=5)
        dt = time.monotonic() - t0
    finally:
        st_p, body_p = held.release()
    d1, m1 = files_of(on, ds)
    text = body_h.decode("utf-8", "replace")
    m = STUCK.fullmatch(text)
    print(f"  {tag}: skip check held in the lock; /health {st_h} {text!r} in {dt:.2f} s; PATCH {st_p} {body_p!r}; "
          f"files {'unchanged' if (d1, m1) == (d0, m0) else 'changed'}")
    if st_h != 503 or m is None or int(m.group(2)) < 0:
        P.append(f"{tag}: /health answered {st_h} {text!r}; want 503 with write_held_ms >= 0")
    if (st_p, body_p) != (200, b"done.") or (d1, m1) != (d0, m0):
        P.append(f"{tag}: the PATCH answered {st_p} {body_p!r} and the files "
                 f"{'did not change' if (d1, m1) == (d0, m0) else 'changed'}; want 200 with the chunk left as stored "
                 "(else the hold was not in the skip check)")


def case_s19h(on, P, rounds):
    tag = "s19h"
    nx, ny, nz = GRID
    clients = nx * nz
    resent = (rounds - 1) * (nx // 2) * nz * ny
    check = counted(on, P, tag, rounds * clients * ny, resent)
    results, problems = {}, []
    harness.sc_concurrent_patch(on, results, tag, rounds=rounds, problems=problems, resend_half=True)
    for key, r in results.items():
        print(f"  {key}: {r['status']}: {r['text'] if r['sha256'] is None else r['sha256']}")
    P += [f"{tag}: {p}" for p in problems]
    check()


def case_settings(image, platform, work, on, off, P):
    on_log, off_log = log_lines(on), log_lines(off)
    startup = "Unchanged chunks covered whole by a write: left as stored (SKIP_UNCHANGED_CHUNKS=1)"
    if startup not in on_log:
        P.append(f"settings: the server with the switch on did not log {startup!r}")
    if not any("Chunks left as stored because a write did not change them:" in x for x in on_log):
        P.append("settings: the server with the switch on logged no line of chunks left as stored")
    st, body = off.request("GET", "/performance")
    noisy = [x for x in off_log if "SKIP_UNCHANGED" in x or "left as stored" in x]
    print(f"  settings: off: /performance counter line {'present' if b'Chunks covered whole' in body else 'absent'}; "
          f"log lines of the switch {noisy}")
    if b"Chunks covered whole" in body or noisy:
        P.append(f"settings: off shows the counter line or logs the switch ({noisy})")
    for value, ignored in (("0", False), ("yes", True)):
        data = os.path.join(work, f"setting-{value}")
        fixtures.segmentation(os.path.join(data, "a_pre"), SIZE, SIZE, harness.RES, prefill_seed=5)
        server = harness.Server(f"skip-setting{value}", image, platform, data, {"SKIP_UNCHANGED_CHUNKS": value})
        try:
            server.start()
            st, vol = read(server, "a_pre", WHOLE, SIZE[::-1])
            d0, _ = files_of(server, "a_pre")
            st2, _ = server.request("PATCH", wpath("a_pre", WHOLE), vol.tobytes())
            d1, _ = files_of(server, "a_pre")
            st3, perf = server.request("GET", "/performance")
            lines = [x for x in log_lines(server) if "SKIP_UNCHANGED" in x or "left as stored" in x]
        finally:
            server.stop()
            server.remove()
        print(f"  settings: SKIP_UNCHANGED_CHUNKS={value}: identical PATCH {st2}, .data {len(d0)} -> {len(d1)} B; "
              f"log {lines}")
        want = [f"SKIP_UNCHANGED_CHUNKS ignored (not 0 or 1): {value}"] if ignored else []
        if lines != want:
            P.append(f"settings: SKIP_UNCHANGED_CHUNKS={value}: log lines {lines}, want {want}")
        if st2 != 200 or len(d1) <= len(d0) or b"Chunks covered whole" in perf:
            P.append(f"settings: SKIP_UNCHANGED_CHUNKS={value} did not behave as off")


def build(data, on):
    """The layers each case uses, written before the server starts."""
    pre = lambda ds, seed: fixtures.segmentation(os.path.join(data, ds), SIZE, SIZE, harness.RES, prefill_seed=seed)
    empty = lambda ds: fixtures.segmentation(os.path.join(data, ds), SIZE, SIZE, harness.RES)
    pre("a_pre", 5)
    empty("a_empty")
    pre("c_pre", 8)
    empty("e1_nodata")
    empty("e1_other")
    if not on:
        return
    pre("b_pre", 6)
    pre("d1_pre", 9)
    empty("d1b_empty")
    pre("d2_pre", 10)
    pre("e2_corrupt", 11)
    pre("e2b_corrupt", 14)
    pre("e2b_twin", 14)
    pre("e4_video", 12)
    pre("e5_pre", 13)
    for ds in ("e2_corrupt", "e2b_corrupt"):
        root = os.path.join(data, ds)
        with open(os.path.join(root, "meta", f"{NAME}.meta"), "rb") as f:
            f.seek(struct.calcsize(fixtures.SHARD_HEADER_LAYOUT))  # chunk 0
            offset, _ = struct.unpack(fixtures.SHARD_LINE_LAYOUT, f.read(12))
        with open(os.path.join(root, "data", f"{NAME}.data"), "r+b") as f:
            f.seek(offset)
            f.write(b"\0\0\0\0")  # its zstd magic number
    with open(os.path.join(data, "e4_video", "meta", f"{NAME}.meta"), "r+b") as f:
        f.seek(6)  # version, dtype, channels, compression
        f.write(struct.pack("<H", 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--rounds", type=int, default=harness.CPATCH_ROUNDS)
    ap.add_argument("--require-hook", action="store_true", help="fail when the build has no test hook")
    args = ap.parse_args()

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    P, skipped = [], []
    data_on, data_off = os.path.join(work, "on"), os.path.join(work, "off")
    build(data_on, True)
    build(data_off, False)
    on = harness.Server("skip-on", args.image, args.platform, data_on, {"SKIP_UNCHANGED_CHUNKS": "1"})
    off = harness.Server("skip-off", args.image, args.platform, data_off)
    cases = [
        ("a", lambda: case_a(on, off, P)),
        ("b", lambda: case_b(on, P)),
        ("c", lambda: case_c(on, off, P)),
        ("d1", lambda: case_d1(on, P, skipped, "d1_pre", "d1")),
        ("d1b", lambda: case_d1(on, P, skipped, "d1b_empty", "d1b")),
        ("d2", lambda: case_d2(on, P, skipped)),
        ("e1", lambda: case_e1(on, off, P)),
        ("e2", lambda: case_e2(on, P)),
        ("e2b", lambda: case_e2b(on, P)),
        ("e4", lambda: case_e4(on, P)),
        ("e5", lambda: case_e5(on, P, skipped)),
        ("s19h", lambda: case_s19h(on, P, args.rounds)),
        ("settings", lambda: case_settings(args.image, args.platform, work, on, off, P)),
    ]
    verdicts = []
    try:
        on.start()
        off.start()
        for name, run in cases:
            before = len(P)
            print(f"case {name}:")
            try:
                run()
            except Exception as e:
                P.append(f"{name}: raised {e!r}")
            for s in (on, off):
                alive, code = s.running()
                if not alive:
                    P.append(f"{name}: the {s.role} server died (exit {code})")
                    s.restart()
            verdicts.append((name, "SKIPPED" if any(x.startswith(name) for x in skipped) and len(P) == before
                             else "PASS" if len(P) == before else "FAIL"))
    finally:
        for s in (on, off):
            s.stop()
            s.save_logs(os.path.join(work, f"{s.role}.log"))
            s.remove()
    # An ASan build: a report that did not kill the server shows only here
    for s in (on, off):
        with open(os.path.join(work, f"{s.role}.log"), errors="replace") as f:
            text = f.read()
        if "ERROR: AddressSanitizer" in text:
            P.append(f"the {s.role} server's log has an AddressSanitizer report")
        sites = sorted({x.split(": runtime error:")[0].rsplit("/", 1)[-1]
                        for x in text.splitlines() if ": runtime error:" in x})
        print(f"{s.role} log: {text.count('ERROR: AddressSanitizer')} AddressSanitizer reports, "
              f"UBSan runtime error sites {sites}")
    if args.require_hook and skipped:
        P.append(f"skipped without the test hook, and --require-hook was given: {skipped}")
    print("verdicts: " + ", ".join(f"{n} {v}" for n, v in verdicts))
    for p in P:
        print(f"PROBLEM {p}")
    print(f"RESULT: {len(P)} problems, {len(skipped)} cases SKIPPED")
    return 1 if P else 0


if __name__ == "__main__":
    sys.exit(main())
