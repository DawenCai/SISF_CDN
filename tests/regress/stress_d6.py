"""Stress tests for stale chunks in the global chunk cache (D6).

--mode patch (the default): a viewer's read that decodes a chunk while a
PATCH replaces it can put its decode of the old chunk into the global chunk
cache after the PATCH has dropped that chunk's line; every later read of the
chunk is then served the old voxels until the line is evicted or the chunk
is written again.

The race sits between two points inside the server (a reader's table read
and its cache insert), so a black-box client cannot make it happen on
demand. This mode makes it likely instead: reader processes read one chunk
without pause while a writer PATCHes it with new contents each time
(incompressible, so each PATCH holds the cache lock for a while), and after
each PATCH, and a pause for reads still in flight, the chunk is read twice
more. Either of those reads returning older contents is a stale re-insert.
With --box-edge E (a multiple of 32) the box is E^3 instead, one corner of
the only mchunk, so each PATCH replaces (E/32)^3 whole chunks (with
WRITE_BATCH_CHUNKS, in batches). Every chunk carries its version in its first
voxel; each read is checked chunk by chunk, and every chunk showing an older
version counts as a stale chunk read. With --slow-batch (a build with the test
hooks and WRITE_BATCH_CHUNKS set) each batch waits 5 ms after its append and
before its entries, while the .meta is unchanged and reads go on without
reloading it: a batch that bumped data_gen before writing its entries lets a
read in that wait take the new data_gen with the old entries and put the old
chunks back, which a run without the wait almost never shows. With
--fail-meta-close (the same build and setting) every batch's .meta close
fails: every PATCH must answer 500 "Could not write chunk", and since its
entries reached the file every chunk read after it must still show its
version (the batch must bump data_gen all the same); the server's log must
show the hook and the failed close.

--mode rename: a request keeps an mchunk's .meta and .data open while it
reads the mchunk's chunks, and a descriptor stays on the file it opened when
another file is renamed over it. If the files are replaced that way during a
long read, and another request then notices the new .meta and reloads the
header (dropping the mchunk's cache lines), the long read must not put the
chunks it goes on reading from the old files into the cache; a PATCH that
merged into such a chunk would write the old voxels into the new files.
Each cycle: a long read of the whole mchunk (a projection, so the answer is
small) starts; the two files are replaced by rename with the next version;
a one-voxel read reloads the header; after the long read ends, the chunk it
read last is read, one voxel of that chunk is PATCHed (so the rest of it is
merged), and the chunk is read again. Either read showing anything but the
new version is stale. A cycle counts as landed when the long read was still
running well after the reload (by --margin seconds), i.e. it read chunks
after it. Harness scenario s15 is one such cycle.

--mode rename-write: the same replacement while a long PATCH is under way:
one that covers every chunk of the mchunk except for a one-voxel border, so
it reads and merges every chunk on the box's faces before it writes. The
rename comes at a later point in each cycle (--delay, then --sweep spread
over the cycles), so it lands before, during and after the PATCH's reads and
writes. Afterwards the whole mchunk is read: outside the box only the new
version may show, and inside only the new version or the PATCH's value. A
PATCH that the replacement interrupted may answer 500, and must then have left
the new files exactly as they were renamed in: the staged files are hashed
before the PATCH starts (the rename moves those same files in, so nothing the
PATCH does can reach them earlier) and the live files again once it answered.
With --sweep auto one PATCH of the box is timed first, with no rename (the
live version is then put back), and the renames come from 0.1 to 0.9 of that
time, each started earlier by the time a docker exec takes (measured too, as
the rename runs in one). A PATCH that answered 500 "Could not write chunk" had the rename land
during its write-back; with --require-mid-write a run with none fails, since
it tested nothing there.

The copies and renames run inside the server's container: a file shared into
a container from the host can be read with its old contents for a moment
after a rename on the host, which would look like the stale reads these
modes look for.

In every mode, before the container is removed, /performance must show the
batched chunk writes line when --env sets WRITE_BATCH_CHUNKS to anything but
0 (with batches counted, and in patch mode chunks in them) and must not show
it otherwise, and the server's log, saved as server.log in the work folder,
must have no AddressSanitizer report.

    python stress_d6.py --image IMAGE --work DIR [--platform linux/arm64]
                        [--mode patch] [--writes 300] [--readers 8]
                        [--box-edge 32] [--env KEY=VALUE ...] [--slow-batch]
                        [--fail-meta-close]
    python stress_d6.py --image IMAGE --work DIR --mode rename [--cycles 20]
                        [--chunk 8] [--delay 0.05]
    python stress_d6.py --image IMAGE --work DIR --mode rename-write
                        [--cycles 12] [--delay 0.05] [--sweep 5.5 | --sweep auto]
                        [--require-mid-write]

--env starts the server with that environment variable (repeatable, any
mode). Exit status 1 if any stale read was seen or a check above failed. A
run that sees no stale read proves nothing on its own: run the same settings
against an image without the fix first, and rely on them only if that run
does see stale reads.
"""

import argparse
import http.client
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

import fixtures

DS = "d6_seg"
RES = (650, 650, 1500)

RENAME_DS = "d6_rename"
RENAME_SIZE = (512, 512, 128)
MCHUNK_NAME = "chunk_0_0_0.0.1X"

# Logged once by a build with the test hooks when --slow-batch slows its batches
SLOW_BATCH_LINE = "TEST HOOK: batches wait 5 ms after their append"

# With --fail-meta-close: what every PATCH must answer, what the hook logs for
# each failed close, and the line the server logs for it (which also holds
# CLOSE_FAILED_NOTE)
CLOSE_FAIL_ANSWER = b"500 Internal Server Error -- Could not write chunk\n"
CLOSE_FAIL_HOOK_LINE = "TEST HOOK: .meta close failed on purpose"
CLOSE_FAILED_LINE = "Metadata entry write failed: "
CLOSE_FAILED_NOTE = "(closing the file)"

# What /performance shows while WRITE_BATCH_CHUNKS is on (the hold is in ms)
BATCH_LINE = re.compile(r"<p>Batched chunk writes \(up to (\d+) chunks per batch\): batches (\d+); "
                        r"chunks in them (\d+); longest hold of the chunk cache lock (\d+(?:\.\d+)?) ms</p>")


def box(edge):
    """The patch mode box: from the origin, so with the dataset's mchunk of
    (2 edge, 2 edge, edge) it is one corner of it and every chunk it touches
    is covered whole, so a PATCH never reads one first. 0-32_0-32_0-32 is
    chunk 0 alone."""
    return f"0-{edge}_0-{edge}_0-{edge}"


def contents(v, edge=32):
    """Box contents for version v: noise, with v in the first voxel of every
    32^3 chunk. The box is x fastest in both the request and the answer, so
    the voxel at x, y, z = 0 comes first, and at edge 32 only that one is v."""
    a = np.random.default_rng(v).integers(0, 65536, size=edge ** 3, dtype=np.uint16)
    a.reshape(edge, edge, edge)[::32, ::32, ::32] = v
    return a


def get_versions(conn, edge):
    """Reads the box; returns each chunk's version (from its first voxel),
    and raises if a chunk's contents are not that version's (a torn chunk)."""
    conn.request("GET", f"/{DS}/1/{box(edge)}")
    r = conn.getresponse()
    body = r.read()
    if r.status != 200 or len(body) != 2 * edge ** 3:
        raise RuntimeError(f"read answered {r.status} with {len(body)} bytes")
    a = np.frombuffer(body, dtype=np.uint16).reshape(edge, edge, edge)  # z, y, x
    versions = a[::32, ::32, ::32]
    for v in np.unique(versions):
        if not v:
            continue
        want = contents(int(v), edge).reshape(edge, edge, edge)
        for cz, cy, cx in np.argwhere(versions == v) * 32:
            c = np.s_[cz:cz + 32, cy:cy + 32, cx:cx + 32]
            if not np.array_equal(a[c], want[c]):
                where = f" in the chunk at x, y, z = {cx}, {cy}, {cz}" if versions.size > 1 else ""
                raise RuntimeError(f"read returned version {v}{where} with contents that are not that version's")
    return [int(v) for v in versions.reshape(-1)]


def reader(port, path, stop, count, errors):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    n = 0
    try:
        while not stop.is_set():
            conn.request("GET", path)
            r = conn.getresponse()
            r.read()
            n += 1
    except Exception as e:  # noqa: BLE001 - reported by the parent
        errors.put(str(e))
    with count.get_lock():
        count.value += n


def start_server(args, work, env):
    """Starts the image on work; returns (container name, port)."""
    name = f"cdn-stress-d6-{os.getpid()}"
    cmd = ["docker", "run", "-d", "--pull", "never", "--name", name, "-p", "127.0.0.1::6000",
           "-v", f"{work}:/data"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    if args.platform:
        cmd += ["--platform", args.platform]
    subprocess.run(cmd + [args.image], check=True, capture_output=True)
    try:
        out = subprocess.run(["docker", "port", name, "6000/tcp"], check=True, capture_output=True, text=True).stdout
        port = int(out.strip().splitlines()[0].rsplit(":", 1)[1])
        deadline = time.time() + 120
        while True:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.request("GET", "/version")
                if c.getresponse().status == 200:
                    break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)
    except BaseException:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise
    return name, port


def check_server(name, port, work, env, need_chunks):
    """Run before the container is removed. /performance must show the
    batched chunk writes line if env turns WRITE_BATCH_CHUNKS on (anything but
    unset, empty or 0, so a value the server ignores fails here), with batches
    counted and, if need_chunks, chunks in them, and must not show it
    otherwise. The server's log is saved to work/server.log and must have no
    AddressSanitizer report. Returns (lines to print, number of problems)."""
    lines, problems = [], 0
    value = env.get("WRITE_BATCH_CHUNKS", "")
    on = re.fullmatch(r"0*", value) is None
    setting = f"WRITE_BATCH_CHUNKS={value}" if "WRITE_BATCH_CHUNKS" in env else "WRITE_BATCH_CHUNKS unset"
    try:
        st, body = request(port, "GET", "/performance")
    except Exception as e:  # noqa: BLE001 - reported below
        st, body = f"no answer ({e})", b""
    text = body.decode("utf-8", "replace") if st == 200 else ""
    m = BATCH_LINE.search(text)
    # Off, not even a line in another form may show
    shown = "Batched chunk writes" in text
    if st != 200:
        lines.append(f"  PROBLEM /performance answered {st}")
        problems += 1
    elif shown and not on:
        lines.append(f"  PROBLEM /performance has a batched chunk writes line with {setting}")
        problems += 1
    elif not on:
        lines.append(f"  OK /performance has no batched chunk writes line with {setting}")
    elif m is None:
        form = " in the expected form" if shown else ""
        lines.append(f"  PROBLEM /performance has no batched chunk writes line{form} with {setting}")
        problems += 1
    else:
        up_to, batches, chunks = (int(m.group(i)) for i in (1, 2, 3))
        seen = f"up to {up_to}, {batches} batches, {chunks} chunks in them, longest hold {m.group(4)} ms"
        if value.isdigit() and up_to != int(value):
            lines.append(f"  PROBLEM /performance batched chunk writes line says {seen}, with {setting}")
            problems += 1
        elif batches == 0 or (need_chunks and chunks == 0):
            lines.append(f"  PROBLEM /performance batched chunk writes line counted nothing: {seen}")
            problems += 1
        else:
            lines.append(f"  OK /performance batched chunk writes: {seen}")

    log = os.path.join(work, "server.log")
    with open(log, "w") as f:
        got = subprocess.run(["docker", "logs", name], stdout=f, stderr=subprocess.STDOUT).returncode
    with open(log, errors="replace") as f:
        asan = f.read().count("ERROR: AddressSanitizer")
    # A log that could not be read cannot show a report: that is not a pass
    if got != 0:
        lines.append(f"  PROBLEM docker logs exited {got}, so the server's log could not be checked ({log})")
        problems += 1
    elif asan:
        lines.append(f"  PROBLEM the server's log has {asan} AddressSanitizer reports ({log})")
        problems += 1
    else:
        lines.append(f"  OK the server's log has no AddressSanitizer report ({log})")
    return lines, problems


def run_patch(args, work, env):
    e = args.box_edge
    path = f"/{DS}/1/{box(e)}"
    fixtures.segmentation(os.path.join(work, DS), (2 * e, 2 * e, e), (2 * e, 2 * e, e), RES)
    name, port = start_server(args, work, env)
    stale, procs, checks, failed = [], [], [], 0
    sent, wrong = 0, None
    stop, count, errors = mp.Event(), mp.Value("q", 0), mp.Queue()
    try:
        if args.slow_batch:
            in_container(name, "touch /data/.test_slow_batch")
        if args.fail_meta_close:
            in_container(name, "touch /data/.test_fail_meta_close")
        procs = [mp.Process(target=reader, args=(port, path, stop, count, errors)) for _ in range(args.readers)]
        for p in procs:
            p.start()

        wconn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        cconn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        t0 = time.time()
        for n in range(args.writes):
            v = (n % 60000) + 1
            wconn.request("PATCH", f"/{DS}+token={fixtures.TOKEN}/write/1/{box(e)}", contents(v, e).tobytes())
            r = wconn.getresponse()
            answered = r.read()
            sent = n + 1
            if args.fail_meta_close:
                # A 500 here still wrote its entries, so the reads below check it
                if r.status != 500 or answered != CLOSE_FAIL_ANSWER:
                    wrong = (n, r.status, answered)
                    break
            elif r.status != 200:
                raise RuntimeError(f"PATCH {n} answered {r.status}")
            time.sleep(args.settle)
            got = [get_versions(cconn, e), get_versions(cconn, e)]
            if any(g != v for read in got for g in read):
                stale.append((n, v, got))
            if not errors.empty():
                break
        elapsed = time.time() - t0
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=30)
        checks, failed = check_server(name, port, work, env, need_chunks=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if args.slow_batch:
        # The hook says so once; without it the run tested nothing slowed
        with open(os.path.join(work, "server.log"), errors="replace") as f:
            slowed = SLOW_BATCH_LINE in f.read()
        if slowed:
            checks.append("  OK the batches waited after their append (--slow-batch)")
        else:
            checks.append(f"  PROBLEM --slow-batch: the server's log has no '{SLOW_BATCH_LINE}' line "
                          f"(a build without the test hooks, or no batch ran)")
            failed += 1
    if args.fail_meta_close:
        if wrong is None:
            checks.append(f"  OK every PATCH answered {CLOSE_FAIL_ANSWER!r} (--fail-meta-close)")
        else:
            checks.append(f"  PROBLEM --fail-meta-close: PATCH {wrong[0]} answered {wrong[1]} {wrong[2][:80]!r}, not 500 "
                          f"{CLOSE_FAIL_ANSWER!r} (the run stopped there)")
            failed += 1
        # Without the hook the run tested nothing failing
        with open(os.path.join(work, "server.log"), errors="replace") as f:
            log = f.read().splitlines()
        if any(CLOSE_FAIL_HOOK_LINE in line for line in log):
            checks.append("  OK the batches' .meta close failed on purpose (--fail-meta-close)")
        else:
            checks.append(f"  PROBLEM --fail-meta-close: the server's log has no '{CLOSE_FAIL_HOOK_LINE}' line "
                          f"(a build without the test hooks, or no batch ran)")
            failed += 1
        if any(CLOSE_FAILED_LINE in line and CLOSE_FAILED_NOTE in line for line in log):
            checks.append(f"  OK the server logged the failed close ('{CLOSE_FAILED_LINE}... {CLOSE_FAILED_NOTE}')")
        else:
            checks.append(f"  PROBLEM --fail-meta-close: the server's log has no '{CLOSE_FAILED_LINE}' line "
                          f"with '{CLOSE_FAILED_NOTE}'")
            failed += 1

    errs = []
    while not errors.empty():
        errs.append(errors.get())
    stale_chunks = sum(g != v for _, v, got in stale for read in got for g in read)
    print(f"image {args.image}, env {env or 'none'}: {sent} PATCHes of box {box(e)} in {elapsed:.1f} s, "
          f"{count.value} concurrent reads from {args.readers} processes, {len(stale)} PATCHes followed by a "
          f"stale read, {stale_chunks} stale chunk reads")
    for n, v, got in stale[:10]:
        if len(got[0]) == 1:
            print(f"  PATCH {n} wrote version {v}; the two reads after it returned {[read[0] for read in got]}")
        else:
            old = sorted({g for read in got for g in read if g != v})
            print(f"  PATCH {n} wrote version {v}; the two reads after it returned "
                  f"{[sum(g != v for g in read) for read in got]} stale chunks, versions {old}")
    for err in errs:
        print(f"  ERROR {err}")
    for line in checks:
        print(line)
    return 1 if stale or errs or failed else 0


def request(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request(method, path, body)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def answer(vol, x0, x1, y0, y1, z0, z1):
    """What a read of the box returns for vol, indexed [x, y, z] (x fastest)."""
    return np.ascontiguousarray(vol[x0:x1, y0:y1, z0:z1].transpose(2, 1, 0)).reshape(-1)


def in_container(name, script):
    """Runs a shell script inside the server's container; see the module
    docstring for why the files are copied and renamed there."""
    subprocess.run(["docker", "exec", name, "sh", "-c", script], check=True, capture_output=True)


def hashes_in_container(name, paths):
    """SHA-256 of each file, read inside the server's container, as above."""
    out = subprocess.run(["docker", "exec", name, "sha256sum", *paths], check=True, capture_output=True,
                         text=True).stdout
    return [line.split()[0] for line in out.splitlines()]


def build_rename_dataset(args, work):
    """The mchunk at three versions (same header, other voxels) under
    /data/versions, version 0 live; returns (versions, live paths in the
    container)."""
    c = args.chunk
    out = os.path.join(work, RENAME_DS)
    fixtures.write_metadata(out, 1, RENAME_SIZE, RES, RENAME_SIZE)
    with open(f"{out}/.sisf_access", "w") as f:
        f.write(fixtures.TOKEN + "\n")
    # Three, so that what a stale read returns can never equal the version
    # just put in place
    versions = [fixtures.pattern(RENAME_SIZE, seed) for seed in (1, 2, 3)]
    # Outside any dataset, but on the same filesystem as the live files, so a
    # rename replaces them in one step
    os.makedirs(os.path.join(work, "versions"))
    os.makedirs(os.path.join(work, "stage"))
    for i, vol in enumerate(versions):
        fixtures.create_shard(f"{work}/versions/{i}.data", f"{work}/versions/{i}.meta", vol, (c, c, c))
    for ext in ("data", "meta"):
        shutil.copyfile(f"{work}/versions/0.{ext}", f"{out}/{ext}/{MCHUNK_NAME}.{ext}")
    return versions, {ext: f"/data/{RENAME_DS}/{ext}/{MCHUNK_NAME}.{ext}" for ext in ("data", "meta")}


def stage_version(name, v):
    in_container(name, f"cp /data/versions/{v}.data /data/stage/next.data && cp /data/versions/{v}.meta /data/stage/next.meta")


def replace_live(name, live):
    in_container(name, f"mv -f /data/stage/next.data {live['data']} && mv -f /data/stage/next.meta {live['meta']}")


def run_rename(args, work, env):
    X, Y, Z = RENAME_SIZE
    c = args.chunk
    versions, live = build_rename_dataset(args, work)

    last = (X - c, X, Y - c, Y, Z - c, Z)  # the chunk the long read reads last
    edit = (X - c, X - c + 1, Y - c, Y - c + 1, Z - c, Z - c + 1)
    edit_path = f"/{RENAME_DS}+token={fixtures.TOKEN}/write/1/" + "{}-{}_{}-{}_{}-{}".format(*edit)
    last_path = f"/{RENAME_DS}/1/" + "{}-{}_{}-{}_{}-{}".format(*last)
    long_path = f"/{RENAME_DS}+project={Z}/1/0-{X}_0-{Y}_0-1"

    name, port = start_server(args, work, env)
    rows, checks, failed = [], [], 0
    try:
        st, _ = request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")  # builds the reader
        if st != 200:
            raise RuntimeError(f"first read answered {st}")
        for k in range(args.cycles):
            old, new = versions[k % 3], versions[(k + 1) % 3]
            stage_version(name, (k + 1) % 3)
            res = {}

            def long_read():
                res["R"] = request(port, "GET", long_path)
                res["R_done"] = time.time()

            th = threading.Thread(target=long_read)
            t0 = time.time()
            th.start()
            time.sleep(args.delay)
            replace_live(name, live)
            st2, _ = request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")
            t_reload = time.time()
            th.join()
            if res["R"][0] != 200 or st2 != 200:
                raise RuntimeError(f"cycle {k}: long read {res['R'][0]}, one-voxel read {st2}")
            landed = res["R_done"] - t_reload > args.margin

            def which(body, vol_new, vol_old, skip_first=False):
                a = np.frombuffer(body, dtype=np.uint16)
                n_, o_ = answer(vol_new, *last), answer(vol_old, *last)
                if skip_first:
                    a, n_, o_ = a[1:], n_[1:], o_[1:]
                return "new" if np.array_equal(a, n_) else ("old" if np.array_equal(a, o_) else "neither")

            st3, b3 = request(port, "GET", last_path)
            read_after = which(b3, new, old) if st3 == 200 else f"status {st3}"
            stp, _ = request(port, "PATCH", edit_path, np.array([7], dtype=np.uint16).tobytes())
            st4, b4 = request(port, "GET", last_path)
            edited = int(np.frombuffer(b4, dtype=np.uint16)[0]) if st4 == 200 else None
            written = which(b4, new, old, skip_first=True) if st4 == 200 else f"status {st4}"
            rows.append({"cycle": k, "landed": landed, "long_read_s": round(res["R_done"] - t0, 3),
                         "after_reload_s": round(res["R_done"] - t_reload, 3), "read_after": read_after,
                         "patch": stp, "edited_voxel": edited, "written": written})
    finally:
        checks, failed = check_server(name, port, work, env, need_chunks=False)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    landed = [r for r in rows if r["landed"]]
    stale_read = [r for r in rows if r["read_after"] != "new"]
    stale_write = [r for r in rows if r["written"] != "new" or r["patch"] != 200 or r["edited_voxel"] != 7]
    print(f"image {args.image}, env {env or 'none'}: {len(rows)} cycles of {args.chunk}^3 chunks, {len(landed)} landed "
          f"(long read still running {args.margin} s after the reload); "
          f"{len(stale_read)} cycles with a stale read, {len(stale_write)} with a stale or failed PATCH")
    for r in rows:
        print(f"  {r}")
    for line in checks:
        print(line)
    return 1 if stale_read or stale_write or failed else 0


def run_rename_write(args, work, env):
    X, Y, Z = RENAME_SIZE
    versions, live = build_rename_dataset(args, work)

    # Every chunk on the box's faces is covered only in part, so the PATCH
    # reads and merges all of them before it writes anything
    b = (1, X - 1, 1, Y - 1, 1, Z - 1)
    inside = np.zeros(RENAME_SIZE, dtype=bool)
    inside[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = True
    body = np.full((b[5] - b[4], b[3] - b[2], b[1] - b[0]), 9, dtype=np.uint16).tobytes()
    patch_path = f"/{RENAME_DS}+token={fixtures.TOKEN}/write/1/" + "{}-{}_{}-{}_{}-{}".format(*b)
    whole_path = f"/{RENAME_DS}/1/0-{X}_0-{Y}_0-{Z}"

    name, port = start_server(args, work, env)
    rows, checks, failed = [], [], 0
    try:
        st, _ = request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")  # builds the reader
        if st != 200:
            raise RuntimeError(f"first read answered {st}")
        start, sweep = args.delay, args.sweep
        if args.sweep == "auto":
            # How long one PATCH takes here, with nothing renamed
            t0 = time.time()
            st, _ = request(port, "PATCH", patch_path, body)
            took = time.time() - t0
            if st != 200:
                raise RuntimeError(f"timing PATCH answered {st}")
            # Version 0 back in place, and its header read, as the cycles start
            stage_version(name, 0)
            replace_live(name, live)
            st, _ = request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")
            if st != 200:
                raise RuntimeError(f"read after the timing PATCH answered {st}")
            # The rename lands a docker exec after its slot; start that much earlier
            execs = []
            for _ in range(3):
                t0 = time.time()
                in_container(name, "true")
                execs.append(time.time() - t0)
            lag = sorted(execs)[1]
            start, sweep = max(0.0, 0.1 * took - lag), 0.8 * took
            print("--sweep auto: one PATCH of box {}-{}_{}-{}_{}-{}".format(*b) + f" took {took:.3f} s with no rename; "
                  f"a docker exec takes {lag:.3f} s; delay {start:.3f} s, sweep {sweep:.3f} s")
        for k in range(args.cycles):
            new = versions[(k + 1) % 3]
            stage_version(name, (k + 1) % 3)
            # The files the rename moves in, before anything can write to them
            staged = hashes_in_container(name, ["/data/stage/next.data", "/data/stage/next.meta"])
            delay = start + sweep * k / max(1, args.cycles - 1)
            res = {}

            def patch():
                res["P"] = request(port, "PATCH", patch_path, body)
                res["P_done"] = time.time()

            th = threading.Thread(target=patch)
            t0 = time.time()
            th.start()
            time.sleep(delay)
            replace_live(name, live)
            t_ren = time.time()
            th.join()
            # A PATCH that answered 500 must not have written into them
            kept = hashes_in_container(name, [live["data"], live["meta"]]) == staged
            request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")  # reloads the header if nothing has
            st, whole = request(port, "GET", whole_path)
            if st != 200:
                raise RuntimeError(f"cycle {k}: whole read answered {st}")
            got = np.frombuffer(whole, dtype=np.uint16).reshape(Z, Y, X).transpose(2, 1, 0)
            # Outside the box only the new files' voxels may show; inside, those
            # or the PATCH's
            bad_out = int(((got != new) & ~inside).sum())
            bad_in = int(((got != new) & (got != 9) & inside).sum())
            painted = int(((got == 9) & inside & (new != 9)).sum())
            rows.append({"cycle": k, "rename_at_s": round(t_ren - t0, 3), "patch_s": round(res["P_done"] - t0, 3),
                         "patch": res["P"][0], "answer": res["P"][1][:80].decode("utf-8", "replace"),
                         "painted_in_new_files": painted, "new_files_unchanged": kept,
                         "stale_outside": bad_out, "stale_inside": bad_in})
    finally:
        checks, failed = check_server(name, port, work, env, need_chunks=False)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    stale = [r for r in rows if r["stale_outside"] or r["stale_inside"] or r["patch"] not in (200, 500)]
    # A PATCH that answered 500 left the new files untouched
    touched = [r for r in rows if r["patch"] == 500 and not r["new_files_unchanged"]]
    during = [r for r in rows if r["rename_at_s"] < r["patch_s"]]
    # The rename landed during the write-back
    in_write = [r for r in rows if r["patch"] == 500 and "Could not write chunk" in r["answer"]]
    print(f"image {args.image}, env {env or 'none'}: {len(rows)} cycles of {args.chunk}^3 chunks, {len(during)} with "
          f"the rename before the PATCH answered ({sum(r['patch'] == 500 for r in rows)} PATCHes answered 500, "
          f"{len(in_write)} of them 'Could not write chunk'); {len(stale)} cycles with old voxels written into the "
          f"new files, {len(touched)} with a PATCH that answered 500 and changed them")
    if args.require_mid_write:
        if in_write:
            checks.append(f"  OK {len(in_write)} cycles had the rename land during the write-back (--require-mid-write)")
        else:
            checks.append("  PROBLEM --require-mid-write: no PATCH answered 500 'Could not write chunk', so no rename "
                          "landed during a write-back and the run tested nothing there")
            failed += 1
    for r in rows:
        print(f"  {r}")
    for line in checks:
        print(line)
    return 1 if stale or touched or failed else 0


def sweep_arg(text):
    """--sweep: seconds, or auto (rename-write mode times one PATCH first)."""
    if text == "auto":
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"takes seconds or auto, not {text!r}") from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", choices=("patch", "rename", "rename-write"), default="patch")
    ap.add_argument("--writes", type=int, default=300, help="patch mode: PATCHes")
    ap.add_argument("--readers", type=int, default=8, help="patch mode: reader processes")
    ap.add_argument("--settle", type=float, default=0.03, help="patch mode: pause after each PATCH before checking, in s")
    ap.add_argument("--box-edge", type=int, default=32,
                    help="patch mode: edge of the box PATCHed and read, a multiple of 32 (32 is one chunk)")
    ap.add_argument("--cycles", type=int, default=20, help="rename mode: replacements")
    ap.add_argument("--chunk", type=int, default=8, help="rename mode: chunk edge; small chunks make the long read long")
    ap.add_argument("--delay", type=float, default=0.05, help="rename mode: from the long read's start to the rename, in s")
    ap.add_argument("--sweep", type=sweep_arg, default=2.0,
                    help="rename-write mode: the rename comes delay + sweep * k / (cycles - 1) s after the PATCH starts; "
                         "auto times one PATCH first and sets delay and sweep to 0.1 and 0.8 times that")
    ap.add_argument("--require-mid-write", action="store_true",
                    help="rename-write mode: fail if no rename landed during a PATCH's write-back (no PATCH answered "
                         "500 'Could not write chunk')")
    ap.add_argument("--margin", type=float, default=0.2,
                    help="rename mode: how long the long read must go on after the reload for the cycle to count as landed, in s")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="environment variable for the server (repeatable)")
    ap.add_argument("--slow-batch", action="store_true",
                    help="patch mode with WRITE_BATCH_CHUNKS: each batch waits 5 ms after its append, before its "
                         "entries (needs a build with the test hooks)")
    ap.add_argument("--fail-meta-close", action="store_true",
                    help="patch mode with WRITE_BATCH_CHUNKS: every batch's .meta close fails, so every PATCH must "
                         "answer 500 and every chunk read after it must still be new (needs a build with the test hooks)")
    args = ap.parse_args()
    env = {}
    for kv in args.env:
        key, sep, value = kv.partition("=")
        if not sep or not key:
            ap.error(f"--env takes KEY=VALUE, not {kv!r}")
        env[key] = value
    if args.mode == "patch":
        e = args.box_edge
        # The cache size the server will use (it ignores a value out of range)
        lines = env.get("CHUNK_CACHE_LINES", "")
        lines = int(lines) if re.fullmatch(r"[0-9]+", lines) and 0 < int(lines) <= 1000000 else 100
        if e <= 0 or e % 32:
            ap.error(f"--box-edge must be a positive multiple of 32, not {e}")
        if (e // 32) ** 3 >= lines:
            ap.error(f"--box-edge {e} covers {(e // 32) ** 3} chunks; "
                     f"it must be fewer than the chunk cache's {lines} lines")
    batched = args.mode == "patch" and re.fullmatch(r"[0-9]*[1-9][0-9]*", env.get("WRITE_BATCH_CHUNKS", "")) is not None
    if args.slow_batch and not batched:
        ap.error("--slow-batch is for --mode patch with WRITE_BATCH_CHUNKS above 0")
    if args.fail_meta_close and not batched:
        ap.error("--fail-meta-close is for --mode patch with WRITE_BATCH_CHUNKS above 0")
    if (args.sweep == "auto" or args.require_mid_write) and args.mode != "rename-write":
        ap.error("--sweep auto and --require-mid-write are for --mode rename-write")

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    return {"patch": run_patch, "rename": run_rename, "rename-write": run_rename_write}[args.mode](args, work, env)


if __name__ == "__main__":
    sys.exit(main())
