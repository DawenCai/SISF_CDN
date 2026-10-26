"""Checks SKIP_UNCHANGED_CHUNKS on one image, on fresh servers and data, and
prints every problem.

    python skip_unchanged.py --image IMAGE --work DIR [--platform linux/amd64]
                             [--rounds 40] [--require-hook] [--batch 0,64,3]

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

--batch takes WRITE_BATCH_CHUNKS values, comma-separated (default 0). The
cases above run once with the variable unset (0, always first, as before)
and once more for each other value, each pass on fresh data and servers
under DIR/b<n> with the variable set on both servers; problems and verdicts
name the pass (b0, b64, ...). Batching changes how often a write holds the
chunk cache lock, not which chunks it writes or their bytes, so after the
last pass every stored file of each other pass must equal pass 0's, byte
for byte (except sc_cpatch, whose frame order depends on timing, and the w
datasets). With the variable set:

- batch counters: /performance shows "Batched chunk writes (up to N chunks
  per batch): batches B; chunks in them C; longest hold of the chunk cache
  lock H ms", and B and C move by what each case wrote: a request that
  writes k chunks of one mchunk opens ceil(k/N) batches, one whose chunks
  are all left as stored opens none, and a batch refused before its append,
  or whose append failed, is not counted (its hold still counts in H).
  Both servers log the start-up line. In pass 0 neither server shows the
  line or logs the variable.
- w1, w2, w3, w4 (hook build, on the off server only, where every chunk
  covered is written): a PATCH of other contents into 8 chunks, each read
  first so the cache holds its old voxels. w1, chunk 8's entry write fails
  (/data/.test_fail_entry_write): 500, chunk 8 keeps its entry and old
  voxels, and the other 7, in later batches too, move into the part of
  .data the PATCH appended. w2, the .meta close fails
  (/data/.test_fail_meta_close): 500 with all 8 written, and a "Metadata
  entry write failed: ... (closing the file)" line. In both every chunk
  reads as the files on disk decode, so no cache line of old voxels
  outlived the write. w3, the whole layer is read first, so the cache holds
  every chunk; while a batch waits after its file check
  (/data/.test_hold_in_batch) /health answers 503 with write_held_ms >= 0,
  and the mchunk's files are replaced by another version of the layer
  (same header, other voxels): 500, the files renamed in unchanged, "Chunk
  written to mchunk files replaced during the write" logged, every chunk of
  the layer reads as the new files decode (a write that took the new
  .meta's time as its own would leave it unread, and the old files' cache
  lines served), and H is at least the time the batch was held. w4, every
  append fails (/data/.test_fail_append), the first batch after waiting at
  the lock (/data/.test_hold_write_lock) for a second longer than H so far:
  500, no entry moves, .data does not grow, every chunk reads its old
  voxels as the files decode, B and C do not move while H takes in that
  wait, and the hook's and "Chunk append failed: " lines are logged. Each
  then answers /health 200 and takes a clean PATCH.
- batch settings (pass 0, only when --batch names a value other than 0):
  "0" is off without a log line, "abc" and "65" are off with one
  "WRITE_BATCH_CHUNKS ignored" line, "64" logs the start-up line and writes
  a whole layer in one batch.

A case that raises is recorded as a problem and the next one runs. At the
end every saved server log of every pass is searched for AddressSanitizer
reports (an ASan build), and the UBSan "runtime error:" sites are listed.

Exit status 1 if any case found a problem. Where a PATCH never stops at the
hook the d, e5 and w cases are reported as SKIPPED (a build without the
hook, or, for d, a PATCH with nothing to skip, which case a would also
show), and --require-hook makes that a failure.
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
import zstd

import fixtures
import harness

SIZE = (128, 128, 64)
C = 32
GRID = (4, 4, 2)
NAME = "chunk_0_0_0.0.1X"
WHOLE = harness.box(0, SIZE[0], 0, SIZE[1], 0, SIZE[2])
SKIP_MARKER = "/data/.test_hold_skip_check"
LOCK_MARKER = "/data/.test_hold_write_lock"
BATCH_MARKER = "/data/.test_hold_in_batch"
ENTRY_MARKER = "/data/.test_fail_entry_write"
CLOSE_MARKER = "/data/.test_fail_meta_close"
APPEND_MARKER = "/data/.test_fail_append"
SKIP_HOOK_LINE = "TEST HOOK: write held before its skip check"
LOCK_HOOK_LINE = "TEST HOOK: write holding the chunk cache lock"
BATCH_HOOK_LINE = "TEST HOOK: batch held after its file check"
ENTRY_HOOK_LINE = "TEST HOOK: entry write failed on purpose"
CLOSE_HOOK_LINE = "TEST HOOK: .meta close failed on purpose"
APPEND_HOOK_LINE = "TEST HOOK: append failed on purpose"
PERF = re.compile(r"<p>Chunks covered whole by writes: (\d+); left as stored because unchanged: (\d+)</p>")
BATCH = re.compile(r"<p>Batched chunk writes \(up to (\d+) chunks per batch\): batches (\d+); chunks in them "
                   r"(\d+); longest hold of the chunk cache lock (\d+(?:\.\d+)?) ms</p>")
STUCK = re.compile(r"stuck lock_wait_ms=(\d+) write_held_ms=(-?\d+)\n")
REPLACED = b"500 Internal Server Error -- Could not write chunk\n"
# The w cases: a box of 2x2x2 chunks, written x slowest, z fastest (ids 0-3, 8-11)
W_BOX = harness.box(0, 2 * C, 0, 2 * C, 0, 2 * C)
W_CHUNKS = [(cx, cy, cz) for cx in range(2) for cy in range(2) for cz in range(2)]
W_DATASETS = ["w1_pre", "w2_pre", "w3_pre", "w4_pre"]
# Another version of w3_pre's layer (same header, other voxels), which w3
# renames in over it
W3_NEW = "w3_new"
# Datasets whose stored bytes are not compared across passes
UNCOMPARED = {"sc_cpatch", *W_DATASETS, W3_NEW}


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


def on_disk(data, meta, i):
    """Chunk i's voxels decoded from the files, indexed [z, y, x] as a read
    returns them, or None if its entry does not decode to a chunk."""
    off, n = entries(meta)[i]
    try:
        raw = zstd.ZSTD_uncompress(data[off:off + n])
    except Exception:
        return None
    if len(raw) != C ** 3 * 2:
        return None
    return np.frombuffer(raw, dtype=np.uint16).reshape(C, C, C).transpose(2, 1, 0)  # stored x slowest, z fastest


def log_lines(server):
    r = subprocess.run(["docker", "logs", server.name], capture_output=True, text=True)
    return (r.stdout + r.stderr).splitlines()


def perf_counts(server):
    st, body = server.request("GET", "/performance")
    m = PERF.search(body.decode("utf-8", "replace")) if st == 200 else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def batch_counts(server):
    """(chunks per batch, batches, chunks in them, longest hold in ms) from
    /performance, or None when it has no batch line."""
    st, body = server.request("GET", "/performance")
    m = BATCH.search(body.decode("utf-8", "replace")) if st == 200 else None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))) if m else None


def batch_startup(n):
    return f"Chunk writes: up to {n} chunks of one mchunk per hold of the chunk cache lock (WRITE_BATCH_CHUNKS={n})"


def batch_log(server):
    """The server's log lines of WRITE_BATCH_CHUNKS."""
    return [x for x in log_lines(server) if "WRITE_BATCH_CHUNKS" in x or "Chunk writes:" in x]


def nbatches(chunks, n):
    """Batches a request opens for chunks consecutive chunks of one mchunk
    (none with n 0, where writes are not batched)."""
    return -(-chunks // n) if n else 0


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


def batched(server, P, tag, n):
    """Returns a function check(batches, chunks) that checks the server's
    batch counters moved by that much and returns them as after, or None.
    With n 0 (WRITE_BATCH_CHUNKS unset) it checks nothing; case batch line
    checks there that the counters are not shown."""
    before = batch_counts(server) if n else None

    def check(batches, chunks):
        if not n:
            return None
        after = batch_counts(server)
        if before is None or after is None:
            P.append(f"{tag}: /performance has no batch line (before {before}, after {after})")
            return None
        moved = (after[1] - before[1], after[2] - before[2])
        print(f"  {tag}: batch counters moved by {moved} (batches, chunks)")
        if moved != (batches, chunks):
            P.append(f"{tag}: batch counters moved by {moved} (batches, chunks), want {(batches, chunks)}")
        return after

    return check


def case_a(on, off, P, n):
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
            bcheck = batched(server, P, tag, n)
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
                bcheck(0, 0)  # every chunk left as stored: no batch
                if d1 != d0 or m1 != m0:
                    P.append(f"{tag}: .data {len(d0)} -> {len(d1)} B, .meta "
                             f"{'unchanged' if m1 == m0 else 'changed'}; want both unchanged")
            else:
                bcheck(nbatches(32, n), 32)
                moved = sum(a != b for a, b in zip(entries(m0), entries(m1)))
                if len(d1) <= len(d0) or moved != 32:
                    P.append(f"{tag}: .data {len(d0)} -> {len(d1)} B with {moved} entries moved; want it to "
                             "grow with all 32 (the control: without the switch every chunk covered whole "
                             "is written)")


def case_b(on, P, n):
    ds, tag = "b_pre", "b"
    st, vol = read(on, ds, WHOLE, SIZE[::-1])
    new = vol.copy()
    new[5, 7, 40] += 1  # z 5, y 7, x 40: chunk (1, 0, 0)
    want_id = chunk_id(1, 0, 0)
    d0, m0 = files_of(on, ds)
    check = counted(on, P, tag, 32, 31)
    bcheck = batched(on, P, tag, n)
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
    bcheck(1, 1)


def case_c(on, off, P, n):
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
        bcheck = batched(server, P, tag, n)
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
            bcheck(1, 1)  # only chunk (0, 0, 0); both chunks covered whole are left as stored
        else:
            bcheck(nbatches(3, n), 3)  # chunk (0, 0, 0) and both chunks covered whole
        out[server is on] = vol1
    if out.get(True) is not None and out.get(False) is not None and not np.array_equal(out[True], out[False]):
        P.append("c: the layer reads back differently on and off")


def case_d1(on, P, skipped, ds, tag, n):
    cb = chunk_box(0, 0, 0)
    st, x = read(on, ds, cb, (C, C, C))
    y = x + 1
    bcheck = batched(on, P, tag, n)
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
    bcheck(2, 2)  # B's chunk, then A's, whose entry B moved


def case_d2(on, P, skipped, n):
    ds, tag = "d2_pre", "d2"
    cb = chunk_box(0, 0, 0)
    st, x = read(on, ds, cb, (C, C, C))
    bcheck = batched(on, P, tag, n)
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
    bcheck(0, 0)  # A's batch is refused at its file check, before its append


def case_e1(on, off, P, n):
    ds = "e1_nodata"
    zeros = np.zeros((C, C, C), dtype=np.uint16).tobytes()
    ones = np.ones((C, C, C), dtype=np.uint16).tobytes()
    for server in (on, off):
        tag = f"e1 ({'on' if server is on else 'off'})"
        server.exec("rm", f"/data/{ds}/data/{NAME}.data")
        bcheck = batched(server, P, tag, n)
        st, body = server.request("PATCH", wpath(ds, chunk_box(0, 0, 0)), zeros, timeout=20)
        st2, body2 = server.request("PATCH", wpath("e1_other", chunk_box(1, 1, 1)), ones, timeout=20)
        print(f"  {tag}: zeros into the mchunk with no .data {st} {body!r}; next PATCH elsewhere {st2} {body2!r}")
        if (st, body) != (500, REPLACED):
            P.append(f"{tag}: answered {st} {body!r}; want 500 {REPLACED!r}")
        if (st2, body2) != (200, b"done."):
            P.append(f"{tag}: the next PATCH elsewhere answered {st2} {body2!r}")
        bcheck(1, 1)  # the batch into the mchunk with no .data fails to open it and is not counted


def case_e2(on, P, n):
    ds, tag = "e2_corrupt", "e2"
    cb = chunk_box(0, 0, 0)
    d0, _ = files_of(on, ds)
    bcheck = batched(on, P, tag, n)
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
    bcheck(2, 2)


def case_e2b(on, P, n):
    ds, tag = "e2b_corrupt", "e2b"
    st, vol = read(on, "e2b_twin", WHOLE, SIZE[::-1])  # the same voxels, intact
    d0, m0 = files_of(on, ds)
    check = counted(on, P, tag, 32, 31)
    bcheck = batched(on, P, tag, n)
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
    bcheck(1, 1)


def case_e4(on, P, n):
    ds, tag = "e4_video", "e4"
    d0, _ = files_of(on, ds)
    bcheck = batched(on, P, tag, n)
    st, body = on.request("PATCH", wpath(ds, chunk_box(1, 1, 1)), np.zeros((C, C, C), dtype=np.uint16).tobytes())
    d1, _ = files_of(on, ds)
    print(f"  {tag}: zeros into a chunk of an mchunk whose header names video compression {st} {body!r}; "
          f".data {len(d0)} -> {len(d1)} B")
    if (st, body) != (200, b"done.") or len(d1) <= len(d0):
        P.append(f"{tag}: answered {st} {body!r} with .data {len(d0)} -> {len(d1)} B; want it written")
    bcheck(1, 1)


def case_e5(on, P, skipped, n):
    ds, tag = "e5_pre", "e5"
    cb = chunk_box(0, 0, 0)
    st, x = read(on, ds, cb, (C, C, C))
    d0, m0 = files_of(on, ds)
    bcheck = batched(on, P, tag, n)
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
    bcheck(0, 0)


def case_s19h(on, P, rounds, n):
    tag = "s19h"
    nx, ny, nz = GRID
    clients = nx * nz
    resent = (rounds - 1) * (nx // 2) * nz * ny
    check = counted(on, P, tag, rounds * clients * ny, resent)
    # Every PATCH that is not a resend writes its column of ny chunks
    writes = clients + (rounds - 1) * (clients - (nx // 2) * nz)
    bcheck = batched(on, P, tag, n)
    results, problems = {}, []
    harness.sc_concurrent_patch(on, results, tag, rounds=rounds, problems=problems, resend_half=True)
    for key, r in results.items():
        print(f"  {key}: {r['status']}: {r['text'] if r['sha256'] is None else r['sha256']}")
    P += [f"{tag}: {p}" for p in problems]
    check()
    bcheck(writes * nbatches(ny, n), writes * ny)


def w_chunks(server, ds):
    """Each chunk of the w box as a read returns it (None if the read
    failed), by chunk id."""
    out = {}
    for k in W_CHUNKS:
        st, a = read(server, ds, chunk_box(*k), (C, C, C))
        out[chunk_id(*k)] = a
    return out


def w_part(vol, cx, cy, cz):
    """Chunk (cx, cy, cz) of a read of the w box."""
    return vol[cz * C:cz * C + C, cy * C:cy * C + C, cx * C:cx * C + C]


def w_start(server, ds):
    """Reads every chunk of the w box, so the cache holds its stored voxels,
    and returns them as x, with y: other contents in every voxel."""
    w_chunks(server, ds)
    st, x = read(server, ds, W_BOX, (2 * C,) * 3)
    if st != 200:
        raise RuntimeError(f"the read of the w box answered {st}")
    return x, x + 1


def hook_count(server, hook_line, ds):
    return sum(hook_line in x and f"/{ds}/" in x for x in log_lines(server))


def fired(server, P, skipped, tag, hook_line, ds, before, st, body):
    """True if the server has logged more than before lines of hook_line for
    ds. Otherwise the case is SKIPPED when the PATCH answered 200 (no test
    hook in this build), and a problem when it did not."""
    if hook_count(server, hook_line, ds) > before:
        return True
    if st == 200:
        print(f"  {tag}: SKIPPED: the PATCH answered {st} without the hook firing (no test hook in this build)")
        skipped.append(tag)
    else:
        P.append(f"{tag}: the PATCH answered {st} {body!r} without the hook firing")
    return False


def w_check(server, P, tag, ds, files, x, y, kept):
    """Checks the w box after a PATCH of y that entered no chunk in kept (its
    entry write, or its batch's append, failed): those keep their entries
    and read x; every other chunk
    of the box moved into the part of .data the PATCH appended, no two
    sharing a byte, and reads y; no chunk outside the box moved; and every
    chunk reads as the files on disk decode, which a cache line of x that
    outlived the write would not. files are the (.data, .meta) before."""
    d0, m0 = files
    d1, m1 = files_of(server, ds)
    e0, e1 = entries(m0), entries(m1)
    want = [chunk_id(*k) for k in W_CHUNKS if chunk_id(*k) not in kept]
    moved = [i for i, (a, b) in enumerate(zip(e0, e1)) if a != b]
    spans = sorted((e1[i][0], e1[i][0] + e1[i][1], i) for i in moved)
    outside = [i for a0, a1, i in spans if a0 < len(d0) or a1 > len(d1)]
    shared = [(ia, ib) for (a0, a1, ia), (b0, b1, ib) in zip(spans, spans[1:]) if b0 < a1]
    served = w_chunks(server, ds)
    wrong, stale = [], []
    for k in W_CHUNKS:
        i = chunk_id(*k)
        got, disk = served[i], on_disk(d1, m1, i)
        if got is None or not np.array_equal(got, w_part(x if i in kept else y, *k)):
            wrong.append(i)
        if got is None or disk is None or not np.array_equal(got, disk):
            stale.append(i)
    print(f"  {tag}: .data {len(d0)} -> {len(d1)} B; entries moved {moved}; chunks reading other than want {wrong}; "
          f"chunks reading other than their files decode {stale}")
    if moved != want:
        P.append(f"{tag}: entries moved {moved}, want {want}")
    if outside:
        P.append(f"{tag}: entries of chunks {outside} lie outside the part of .data the PATCH appended "
                 f"({len(d0)}-{len(d1)})")
    if shared:
        P.append(f"{tag}: entries of chunks {shared} share bytes")
    if wrong:
        old = f"their old voxels for {kept}" if kept else ""
        new = "the PATCH's" if len(kept) < len(W_CHUNKS) else ""
        P.append(f"{tag}: chunks {wrong} read back other than {' and '.join(x for x in (old, new) if x)}")
    if stale:
        P.append(f"{tag}: chunks {stale} read back other than their files decode (a cache line of the old voxels "
                 "outlived the write)")


def w_after(server, P, tag, ds, y, n):
    """After the failed PATCH: /health answers 200 and a clean PATCH of y lands."""
    st_h, body_h = server.request("GET", "/health", timeout=5)
    bcheck = batched(server, P, f"{tag} clean PATCH", n)
    st, body = server.request("PATCH", wpath(ds, W_BOX), y.tobytes())
    st2, back = read(server, ds, W_BOX, (2 * C,) * 3)
    print(f"  {tag}: then /health {st_h} {body_h!r}; a clean PATCH {st} {body!r}; read {st2}")
    if st_h != 200:
        P.append(f"{tag}: /health answered {st_h} {body_h!r} after the failed write")
    if (st, body) != (200, b"done.") or st2 != 200 or not np.array_equal(back, y):
        P.append(f"{tag}: a clean PATCH answered {st} {body!r} or the box does not read back as sent ({st2})")
    bcheck(nbatches(8, n), 8)


def case_w1(off, P, skipped, n):
    ds, tag = "w1_pre", "w1"
    fail = chunk_id(1, 0, 0)  # 8: the 5th chunk written, so batches after its own still run
    x, y = w_start(off, ds)
    files = files_of(off, ds)
    before = hook_count(off, ENTRY_HOOK_LINE, ds)
    bcheck = batched(off, P, tag, n)
    off.exec("sh", "-c", f"echo {fail} > {ENTRY_MARKER}")
    try:
        st, body = off.request("PATCH", wpath(ds, W_BOX), y.tobytes())
    finally:
        off.exec("rm", "-f", ENTRY_MARKER)
    if not fired(off, P, skipped, tag, ENTRY_HOOK_LINE, ds, before, st, body):
        return
    print(f"  {tag}: chunk {fail}'s entry write failed on purpose; PATCH {st} {body!r}")
    if (st, body) != (500, REPLACED):
        P.append(f"{tag}: PATCH answered {st} {body!r}; want 500 {REPLACED!r}")
    bcheck(nbatches(8, n), 8)  # a batch whose entry write failed got past its append, so it counts
    w_check(off, P, tag, ds, files, x, y, [fail])
    w_after(off, P, tag, ds, y, n)


def case_w2(off, P, skipped, n):
    ds, tag = "w2_pre", "w2"
    x, y = w_start(off, ds)
    files = files_of(off, ds)
    before = hook_count(off, CLOSE_HOOK_LINE, ds)
    bcheck = batched(off, P, tag, n)
    # The server logs one "Metadata entry write failed" line a second at most,
    # and w1 has just logged one
    time.sleep(1.1)
    off.exec("touch", CLOSE_MARKER)
    try:
        st, body = off.request("PATCH", wpath(ds, W_BOX), y.tobytes())
    finally:
        off.exec("rm", "-f", CLOSE_MARKER)
    if not fired(off, P, skipped, tag, CLOSE_HOOK_LINE, ds, before, st, body):
        return
    logged = [line for line in log_lines(off) if "Metadata entry write failed: " in line and f"/{ds}/" in line
              and "(closing the file)" in line]
    print(f"  {tag}: the .meta close failed on purpose; PATCH {st} {body!r}; log {logged}")
    if (st, body) != (500, REPLACED):
        P.append(f"{tag}: PATCH answered {st} {body!r}; want 500 {REPLACED!r}")
    if not logged:
        P.append(f"{tag}: no 'Metadata entry write failed: ... (closing the file)' line for {ds}")
    bcheck(nbatches(8, n), 8)
    w_check(off, P, tag, ds, files, x, y, [])
    w_after(off, P, tag, ds, y, n)


def case_w3(off, P, skipped, n):
    ds, tag = "w3_pre", "w3"
    x, y = w_start(off, ds)
    # Every chunk of the layer, so the cache holds lines of the old files
    # outside the box too
    read(off, ds, WHOLE, SIZE[::-1])
    new = files_of(off, W3_NEW)
    bcheck = batched(off, P, tag, n)
    held = Held(off, BATCH_MARKER, BATCH_HOOK_LINE, wpath(ds, W_BOX), y.tobytes())
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
        st_h, body_h = off.request("GET", "/health", timeout=5)
        off.exec("sh", "-c", f"cd /data/{ds} && cp ../{W3_NEW}/data/{NAME}.data data/{NAME}.data.new && "
                             f"cp ../{W3_NEW}/meta/{NAME}.meta meta/{NAME}.meta.new && "
                             f"mv data/{NAME}.data.new data/{NAME}.data && mv meta/{NAME}.meta.new meta/{NAME}.meta")
        held_ms = (time.monotonic() - t0) * 1000  # the batch holds the lock for longer than this
    finally:
        st, body = held.release()
    # A request that did not answer comes back as a status word and a str
    text = body_h.decode("utf-8", "replace") if isinstance(body_h, bytes) else str(body_h)
    m = STUCK.fullmatch(text)
    after_files = files_of(off, ds)
    logged = [line for line in log_lines(off) if "Chunk written to mchunk files replaced during the write: " in line
              and f"/{ds}/" in line]
    st2, vol = read(off, ds, WHOLE, SIZE[::-1])
    stale = [chunk_id(cx, cy, cz) for cx in range(GRID[0]) for cy in range(GRID[1]) for cz in range(GRID[2])
             if vol is None or not np.array_equal(w_part(vol, cx, cy, cz), on_disk(*new, chunk_id(cx, cy, cz)))]
    print(f"  {tag}: batch held; /health {st_h} {text!r}; files replaced by another version of the layer, PATCH "
          f"released {st} {body!r}; files {'as renamed in' if after_files == new else 'changed'}; log {logged}; "
          f"layer read {st2}; chunks reading other than the new files decode {stale}")
    if (st, body) != (500, REPLACED):
        P.append(f"{tag}: PATCH answered {st} {body!r}; want 500 {REPLACED!r}, as a write into replaced files")
    if st_h != 503 or m is None or int(m.group(2)) < 0:
        P.append(f"{tag}: /health answered {st_h} {text!r} while the batch held the lock; want 503 with "
                 "write_held_ms >= 0")
    if after_files != new:
        P.append(f"{tag}: the files renamed in changed after the PATCH")
    if not logged:
        P.append(f"{tag}: no 'Chunk written to mchunk files replaced during the write' line for {ds}")
    if st2 != 200 or stale:
        P.append(f"{tag}: chunks {stale} read back other than the files renamed in decode (read {st2}); a cache "
                 "line of the old files outlived the write, as when the .meta renamed in is never reloaded")
    # The held batch got past its append and counts; a later one of the
    # request finds the copies at its file check and does not
    after = bcheck(1, min(n, len(W_CHUNKS)))
    if after is not None:
        print(f"  {tag}: longest hold of the chunk cache lock {after[3]} ms; the batch was held over {held_ms:.0f} ms")
        if after[3] < held_ms:
            P.append(f"{tag}: longest hold of the chunk cache lock {after[3]} ms, less than the {held_ms:.0f} ms "
                     "the batch was held")
    w_after(off, P, tag, ds, y, n)


def case_w4(off, P, skipped, n):
    ds, tag = "w4_pre", "w4"
    x, y = w_start(off, ds)
    files = files_of(off, ds)
    before = hook_count(off, APPEND_HOOK_LINE, ds)
    bcheck = batched(off, P, tag, n)
    # The request's first batch waits at the lock (test_hold_write_lock) for
    # longer than any hold so far, then fails its append: the longest hold
    # must still take it in, although the batch returns before the counters
    longest = (batch_counts(off) or (0, 0, 0, 0.0))[3]
    hold_s = longest / 1000 + 1
    off.exec("touch", APPEND_MARKER)
    try:
        held = Held(off, LOCK_MARKER, LOCK_HOOK_LINE, wpath(ds, W_BOX), y.tobytes())
        t0 = time.monotonic()
        if held.held:
            time.sleep(hold_s)
        held_ms = (time.monotonic() - t0) * 1000  # the batch holds the lock for longer than this
        st, body = held.release()
    finally:
        off.exec("rm", "-f", APPEND_MARKER, LOCK_MARKER)
    if not fired(off, P, skipped, tag, APPEND_HOOK_LINE, ds, before, st, body):
        return
    d1, _ = files_of(off, ds)
    logged = [line for line in log_lines(off) if "Chunk append failed: " in line and f"/{ds}/" in line]
    print(f"  {tag}: every append failed on purpose, the first batch held {held_ms:.0f} ms first; PATCH {st} "
          f"{body!r}; .data {len(files[0])} -> {len(d1)} B; log {logged}")
    if (st, body) != (500, REPLACED):
        P.append(f"{tag}: PATCH answered {st} {body!r}; want 500 {REPLACED!r}")
    if not held.held:
        P.append(f"{tag}: the failing batch never waited at the lock hook")
    if len(d1) != len(files[0]):
        P.append(f"{tag}: .data {len(files[0])} -> {len(d1)} B; want it as it was, with no frame appended")
    if not logged:
        P.append(f"{tag}: no 'Chunk append failed: ' line for {ds}")
    after = bcheck(0, 0)  # every batch of the request failed its append
    if after is not None:
        print(f"  {tag}: longest hold of the chunk cache lock {after[3]} ms (before {longest} ms); the failing batch "
              f"was held over {held_ms:.0f} ms")
        if after[3] < held_ms:
            P.append(f"{tag}: longest hold of the chunk cache lock {after[3]} ms, less than the {held_ms:.0f} ms "
                     "a batch that then failed its append held it")
    w_check(off, P, tag, ds, files, x, y, [chunk_id(*k) for k in W_CHUNKS])
    w_after(off, P, tag, ds, y, n)


def case_batch_line(on, off, P, n):
    for server in (on, off):
        tag = f"batch line ({'on' if server is on else 'off'})"
        lines = batch_log(server)
        st, body = server.request("GET", "/performance")
        text = body.decode("utf-8", "replace")
        shown = [x for x in text.splitlines() if "Batched chunk writes" in x]
        m = BATCH.search(text) if st == 200 else None
        print(f"  {tag}: log lines of the switch {lines}; /performance {st} {shown}")
        want = [batch_startup(n)] if n else []
        if lines != want:
            P.append(f"{tag}: log lines of the switch {lines}, want {want}")
        if n and (m is None or int(m.group(1)) != n):
            P.append(f"{tag}: /performance shows {shown}; want the batch line with up to {n} chunks per batch")
        if not n and shown:
            P.append(f"{tag}: /performance shows {shown} with WRITE_BATCH_CHUNKS unset")


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


def case_batch_settings(image, platform, work, P, logs):
    for value, n in (("0", 0), ("abc", 0), ("65", 0), ("64", 64)):
        data = os.path.join(work, f"batch-setting-{value}")
        log = os.path.join(work, f"batch-setting-{value}.log")
        fixtures.segmentation(os.path.join(data, "a_pre"), SIZE, SIZE, harness.RES, prefill_seed=5)
        server = harness.Server(f"batch-setting{value}", image, platform, data, {"WRITE_BATCH_CHUNKS": value})
        try:
            server.start()
            st, vol = read(server, "a_pre", WHOLE, SIZE[::-1])
            st2, _ = server.request("PATCH", wpath("a_pre", WHOLE), vol.tobytes())
            st3, perf = server.request("GET", "/performance")
            lines = batch_log(server)
        finally:
            server.stop()
            server.save_logs(log)
            server.remove()
            logs.append((f"batch setting {value}", log))
        text = perf.decode("utf-8", "replace")
        shown = [x for x in text.splitlines() if "Batched chunk writes" in x]
        m = BATCH.search(text)
        print(f"  batch settings: WRITE_BATCH_CHUNKS={value}: whole-layer PATCH {st2}; /performance {shown}; "
              f"log {lines}")
        if n:
            want = [batch_startup(n)]
        elif value != "0":
            want = [f"WRITE_BATCH_CHUNKS ignored (not a whole number from 0 to 64): {value}; writes are not batched"]
        else:
            want = []
        if lines != want:
            P.append(f"batch settings: WRITE_BATCH_CHUNKS={value}: log lines {lines}, want {want}")
        if st2 != 200:
            P.append(f"batch settings: WRITE_BATCH_CHUNKS={value}: the PATCH answered {st2}")
        if n and (m is None or (int(m.group(1)), int(m.group(2)), int(m.group(3))) != (n, 1, 32)):
            P.append(f"batch settings: WRITE_BATCH_CHUNKS={value}: /performance shows {shown}; want up to {n} chunks "
                     "per batch and the whole layer, 32 chunks, in one batch")
        if not n and shown:
            P.append(f"batch settings: WRITE_BATCH_CHUNKS={value} did not behave as off: /performance shows {shown}")


def build(data, on):
    """The layers each case uses, written before the server starts."""
    pre = lambda ds, seed: fixtures.segmentation(os.path.join(data, ds), SIZE, SIZE, harness.RES, prefill_seed=seed)
    empty = lambda ds: fixtures.segmentation(os.path.join(data, ds), SIZE, SIZE, harness.RES)
    pre("a_pre", 5)
    empty("a_empty")
    pre("c_pre", 8)
    empty("e1_nodata")
    empty("e1_other")
    for i, ds in enumerate(W_DATASETS):
        pre(ds, 15 + i)
    pre(W3_NEW, 19)
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


def stored(root):
    """SHA-256 of every file under root, by its path below root, except the
    UNCOMPARED datasets and the test hooks' markers."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root)
            if rel.split(os.sep)[0] in UNCOMPARED or (os.sep not in rel and rel.startswith(".test_")):
                continue
            out[rel] = harness.file_sha(path)
    return out


def compare_stored(work, values, P):
    """Every stored file of each pass after the first, on both sides, against
    pass 0's."""
    for n in values[1:]:
        for side in ("on", "off"):
            ref = stored(os.path.join(work, "b0", side))
            got = stored(os.path.join(work, f"b{n}", side))
            missing = sorted(set(ref) - set(got))
            extra = sorted(set(got) - set(ref))
            differ = sorted(p for p in set(ref) & set(got) if ref[p] != got[p])
            print(f"stored bytes: b{n} {side}: {len(got)} files against b0's {len(ref)}; missing {len(missing)}, "
                  f"extra {len(extra)}, differing {len(differ)}")
            for what, paths in (("missing", missing), ("extra", extra), ("differing", differ)):
                by_ds = {}
                for p in paths:
                    ds, _, rest = p.partition(os.sep)
                    by_ds.setdefault(ds, []).append(rest or ds)
                for ds, rels in sorted(by_ds.items()):
                    P.append(f"b{n} stored bytes ({side}): {ds}: {what} against b0: {rels}")


def run_pass(args, root, n, batching, P, skipped, logs):
    """Every case on fresh data and servers under root, with
    WRITE_BATCH_CHUNKS=n on both servers (unset when n is 0). Problems and
    SKIPPED cases go to P and skipped named by the pass, and the servers'
    saved logs to logs. Returns the verdicts."""
    Pp, sk = [], []
    data_on, data_off = os.path.join(root, "on"), os.path.join(root, "off")
    build(data_on, True)
    build(data_off, False)
    batch = {"WRITE_BATCH_CHUNKS": str(n)} if n else {}
    on = harness.Server("skip-on", args.image, args.platform, data_on, {"SKIP_UNCHANGED_CHUNKS": "1", **batch})
    off = harness.Server("skip-off", args.image, args.platform, data_off, batch)
    cases = [
        ("a", lambda: case_a(on, off, Pp, n)),
        ("b", lambda: case_b(on, Pp, n)),
        ("c", lambda: case_c(on, off, Pp, n)),
        ("d1", lambda: case_d1(on, Pp, sk, "d1_pre", "d1", n)),
        ("d1b", lambda: case_d1(on, Pp, sk, "d1b_empty", "d1b", n)),
        ("d2", lambda: case_d2(on, Pp, sk, n)),
        ("e1", lambda: case_e1(on, off, Pp, n)),
        ("e2", lambda: case_e2(on, Pp, n)),
        ("e2b", lambda: case_e2b(on, Pp, n)),
        ("e4", lambda: case_e4(on, Pp, n)),
        ("e5", lambda: case_e5(on, Pp, sk, n)),
        ("s19h", lambda: case_s19h(on, Pp, args.rounds, n)),
    ]
    if n:
        # On the off server, where every chunk covered is written and no
        # skip hook can fire
        cases += [
            ("w1", lambda: case_w1(off, Pp, sk, n)),
            ("w2", lambda: case_w2(off, Pp, sk, n)),
            ("w3", lambda: case_w3(off, Pp, sk, n)),
            ("w4", lambda: case_w4(off, Pp, sk, n)),
        ]
    cases.append(("batch line", lambda: case_batch_line(on, off, Pp, n)))
    if not n:
        cases.append(("settings", lambda: case_settings(args.image, args.platform, root, on, off, Pp)))
        if batching:
            cases.append(("batch settings", lambda: case_batch_settings(args.image, args.platform, root, Pp, logs)))
    verdicts = []
    print(f"pass b{n}: WRITE_BATCH_CHUNKS={n if n else 'unset'}")
    try:
        on.start()
        off.start()
        for name, run in cases:
            before = len(Pp)
            print(f"case {name}:")
            try:
                run()
            except Exception as e:
                Pp.append(f"{name}: raised {e!r}")
            for s in (on, off):
                alive, code = s.running()
                if not alive:
                    Pp.append(f"{name}: the {s.role} server died (exit {code})")
                    s.restart()
            verdicts.append((f"b{n} {name}", "SKIPPED" if any(x.startswith(name) for x in sk) and len(Pp) == before
                             else "PASS" if len(Pp) == before else "FAIL"))
    finally:
        for s in (on, off):
            s.stop()
            s.save_logs(os.path.join(root, f"{s.role}.log"))
            s.remove()
            logs.append((f"b{n} {s.role}", os.path.join(root, f"{s.role}.log")))
    P += [f"b{n} {p}" for p in Pp]
    skipped += [f"b{n} {x}" for x in sk]
    return verdicts


def batch_values(text):
    """--batch: 0 first, the reference pass, then each other value once."""
    out = [0]
    for x in text.split(","):
        if int(x) not in out:
            out.append(int(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--rounds", type=int, default=harness.CPATCH_ROUNDS)
    ap.add_argument("--require-hook", action="store_true", help="fail when the build has no test hook")
    ap.add_argument("--batch", type=batch_values, default="0",
                    help="WRITE_BATCH_CHUNKS values, comma-separated; one pass each, 0 (unset) always first")
    args = ap.parse_args()

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    P, skipped, logs, verdicts = [], [], [], []
    for n in args.batch:
        verdicts.append(run_pass(args, os.path.join(work, f"b{n}"), n, len(args.batch) > 1, P, skipped, logs))
    compare_stored(work, args.batch, P)
    # An ASan build: a report that did not kill the server shows only here
    for label, path in logs:
        with open(path, errors="replace") as f:
            text = f.read()
        if "ERROR: AddressSanitizer" in text:
            P.append(f"the {label} server's log has an AddressSanitizer report")
        sites = sorted({x.split(": runtime error:")[0].rsplit("/", 1)[-1]
                        for x in text.splitlines() if ": runtime error:" in x})
        print(f"{label} log: {text.count('ERROR: AddressSanitizer')} AddressSanitizer reports, "
              f"UBSan runtime error sites {sites}")
    if args.require_hook and skipped:
        P.append(f"skipped without the test hook, and --require-hook was given: {skipped}")
    for v in verdicts:
        print("verdicts: " + ", ".join(f"{n} {s}" for n, s in v))
    for p in P:
        print(f"PROBLEM {p}")
    print(f"RESULT: {len(P)} problems, {len(skipped)} cases SKIPPED")
    return 1 if P else 0


if __name__ == "__main__":
    sys.exit(main())
