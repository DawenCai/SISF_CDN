# Black-box regression harness

Runs two CDN images side by side on the same generated datasets, sends both the
same requests, and reports every response that differs.

```
pip install numpy zstd
python harness.py --baseline tuffr5/sisf_cdn@sha256:83d43adc... --baseline-platform linux/amd64 \
                  --candidate sisf_cdn:candidate --work /tmp/regress \
                  --allow expected_diffs.json --report /tmp/report.json
```

Servers are started with `docker run --pull never` (here and in
`stress_d6.py`), so both images must already be local: `docker pull` the
baseline first. A missing image fails the run instead of being fetched.

- `fixtures.py` writes the datasets. Its bytes match pySISF 0.4.1 and the
  portal's tiled writer (checked file by file when it was written).
- Reads: status code and SHA-256 of every body. Writes: the portal's
  read-merge-PATCH sequence on each server's own copy, then re-reads and a
  byte comparison of every file on disk.
- A server that dies is recorded as CRASH and restarted. Two crashes on the
  same request count as the same behaviour; every crash is listed either way.
  A server that stays up but does not answer in time is recorded as TIMEOUT.
- After the write and danger stages, `scenarios` runs the cases that kill,
  hang or lose data on production, each on a dataset it builds while the
  server runs, with a liveness check after each:
  - s1: stale archive geometry after an in-place re-conversion (the
    2026-08-30 abort), then `/info` and a full read of the same dataset
    afterwards;
  - s2: a corrupt zstd frame; s3: an unknown compression type in an mchunk
    header;
  - s4: a writable layer with a missing `.data` file (production leaks its
    write lock; this case uses a 20 s client timeout and restarts a server
    that timed out);
  - s5: a PATCH over a chunk whose data is cut short (production overwrites
    the chunk's other voxels with zeros and answers 200);
  - s6: the portal's read-merge-PATCH with one mchunk's `.data` unreadable
    during the read only (renamed inside the container with `docker exec`
    and back), plus a viewer's read of the same failure;
  - s7: a tile re-converted in place with a wider tile while the server
    holds readers for it, then read and written in the one chunk whose
    extent grew; it runs last because production dies in it;
  - s8: a `raw_access` read wider than the mchunk's stored tile;
  - s23: `raw_access` on the 3-channel vol3c, once per channel key. The
    answer is sized for every channel but holds only the key's channel, in
    its first block; the rest must be zero. Production allocates it with
    malloc and leaves the rest as it was. A second server is started with
    `MALLOC_PERTURB_=165` (glibc then fills every malloc, not calloc), so a
    build that still uses malloc fails every time, and with the ASan
    allocator's own fill options for the sanitizer build. The record holds
    counts only: channels with a wrong length, a nonzero unfilled part, or a
    first block that is not the image route's channel;
  - s9: s6's sequence with the `.meta` unreadable instead, on an mchunk the
    server has never opened, so no reader can be built for it during the
    read;
  - s10: an mchunk header naming video compression over zstd frames shorter
    than the video header, read by a viewer and by the portal and under a
    PATCH;
  - s11, s12: a dataset whose `metadata.bin` is empty, or names an mchunk
    size of 0 (production divides by it in the inventory scan; the harness
    removes the dataset before restarting a server that died on it);
  - s17: reads sent one after another on one keep-alive connection, as
    the portal's httpx pool and browsers send them: an image read over
    1 MiB, then a mesh file (a route that adds two headers), and the
    reverse; then GETs sent with `Expect: 100-continue` mixed with large
    and small reads, the last with `Connection: close`; then 16 threads,
    each on its own connection, cycling through large and small reads (448
    in all), every body checked against the same read on a fresh
    connection. After a body of 1 MiB or more, production's crow leaves a
    stale `connection` header in the connection's next response, which
    then carries two `Connection` headers, and frees it while that
    response is still being sent. Its `100 Continue` shares the completion
    of a whole response, which clears the real answer while it is still
    queued: the first body byte goes out as 0, and a `Connection: close`
    answer is cut off. The sanitizer build aborts on both. Production's
    `Connection` headers and connection reuse in the `Expect` case change
    from run to run, so there they are kept in one record of their own,
    which the allow file pins for the fork only (one header each, one
    connection), and the baseline is compared by status and body;
  - s18: `skeleton_api` upload, replace and delete (a GET) on a dataset
    with a `traces.sql`, with `ls` and `get` before and after, on this
    server (writes off by default), on a second server with
    `SKELETON_API_WRITES=1` and its own dataset, and on one with a value
    that is not 0 or 1. Rows carry the time they were written, so the
    file's bytes are not compared; each server records whether it changed;
  - s13: `+channel=N` reads of the 3-channel fixture, each checked against
    channel N's block of the same read without the filter (boxes, a plane,
    a coarser level, a projection, a gaussian filter), channels that do
    not exist, and channel items that cannot be parsed (`+channel=`,
    `+channel`, two `=`, a second `+`);
  - s14: a second server on the same data started with `MAX_READ_VOXELS`
    set (a box, per channel): image reads under, at and over the limit on
    the 3-channel and the 1-channel fixture, a projection at and over it, a
    reversed box, one channel of a box over it, `tracing` and `raw_access`
    under and over it, and a server whose limit is not a number;
  - s15: an mchunk's `.meta` and `.data` replaced by rename (inside the
    container) while a long read of it is under way, then read and PATCHed
    after another read reloaded the header: the chunk the long read read
    last, and the PATCH merged into it, must be the new files' (a request
    that keeps its files open must not put the old files' chunks into the
    cache). Every answer compared is the same wherever the rename lands;
    whether it catches a broken build depends on it landing while the long
    read is still reading (see `stress_d6.py --mode rename`);
  - s19: PATCHes sent at once into one writable mchunk (4x4x2 chunks of
    32^3, all appended to the same `.data` and entered in the same
    `.meta`): each of 40 rounds, 8 clients each PATCH a column of 4 whole
    chunks that only they write, with new contents that zstd cannot
    compress, then every chunk is read back with the token. After the last
    round every `.meta` entry must lie inside the `.data`, share no byte
    with another entry and decode to the chunk sent last. A build that
    appends two chunks at once without the chunk lock answers 200 and
    points chunks at each other's bytes. Where the frames land depends on
    which write goes first, so these files are not in the byte comparison.
    `concurrent_patch.py --image IMAGE --work DIR [--runs N]` runs s19 on
    one image and fails on any problem, without a baseline;
  - s20: PATCHes into levels 2 and 4 of a writable layer that has levels 1,
    2 and 4, every level of every mchunk created empty as the portal would
    (production and stage 2 refuse every level but 1): a box across
    mchunks at level 2, one out to level 4's far edges, one inside chunks
    level 2 already holds, then one at level 1. After each, every level is
    read whole with the token and must equal what was sent to it with a
    200, and the level-1 files must be the same bytes on both servers. Then
    level 0, a level the layer does not have (8), a box past level 2's own
    size but inside level 1's, and a level-4 box across an mchunk with no 4X
    files are refused, and every file of both datasets is as it was before
    them. A build that lets the level through the route check but writes
    level 1 answers 200 and puts the level-2 voxels into level 1.
  - s21: image reads whose box runs backwards on x, y or z (on x also with
    a filter), and boxes with a coordinate that does not fit in 32 bits:
    z 19 to 2^32 + 18 on level 2 (18 deep), with and without a gaussian
    filter, z 0 to 2^32 + 1, x 2^32 to 2^32 + 8, and x 2^32 + 2 to 8.
    Production fails a reversed box with 500 and dies on one with a filter;
    it keeps only the low 32 bits of each coordinate, so the large ones
    become a reversed box or a smaller one that it serves. The fork answers
    each of these 400 'Invalid data range'. Controls that answer alike on
    both: a forward box, empty boxes on x, y and z (with and without a
    filter) and a box past the image.
  - s22: coordinates past 32 bits, or negative, on the write route and on
    raw_access. On a writable layer of its own, after a forward PATCH (the
    control): PATCHes of x 2^32 + 8 to 2^32 + 16, x 2^32 + 8 to 16, x 8 to
    2^32 + 16 and x -(2^32 - 8) to 16, which production writes at x 8-16;
    boxes reversed on x and y (2-byte body), on y and on z; and empty boxes
    on y and z, which write nothing on both. The fork refuses every one but
    the empty boxes with 400, and the layer's files and its x 8-16 must be as
    they were. raw_access, on vol1c's mchunk (0,0,0), gets z 0 to 2^32 + 1,
    x 2^32 + 2 to 8 and x -(2^32 - 6) to 8 (production serves z 0-1, x 2-8
    and x 6-8; the fork answers 400), beside a forward and an empty box that
    answer alike on both. The layer is not in the final byte comparison:
    production writes it.
  - s24: the levels of an axis held in ONE mchunk (the portal's writer makes
    x and y one mchunk up to 4096, and z one mchunk while z <= 512 and
    x * y * z <= 2^31). The writer halves the whole
    mchunk level by level (`d // 2`), so level S holds floor(size / S)
    planes there; the CDN used to declare one fewer wherever the size is not
    a multiple of S, which hid the last stored plane and, with the exact
    voxel size, put the viewer's clip bound (size / S) more than a plane past
    the declared size, so its prefetch asked for chunks that start past the
    end and whose size wraps to a URL end past 2^32 (BUG-213). Five
    datasets are built while the server runs: three images with every axis in
    one mchunk (70 x 60 x 13, 64 x 64 x 15, 45 x 50 x 17, levels 1, 2, 4,
    8), one with every axis in two mchunks of 37 (70 x 70 x 70, levels 1, 2,
    4; its sizes must stay as they were),
    and a 2 x 2 tiled one with z in one mchunk and a crop window; plus a
    writable layer (70 x 60 x 13, skeleton levels). Records: each dataset's
    level sizes and voxel sizes (level 1 is the same on both servers); every
    one-mchunk axis size against the shape and crop window in the `.meta`
    header on disk (none may be past what the file holds, and the untiled
    ones must equal it); the last stored plane of each axis at each level
    read through the image route and compared with the fixture; a max
    projection that runs into the last plane; the voxel sizes
    `raw_access/.../info` lists; the chunk
    requests a viewer would make past the declared sizes, worked out from
    `/info` (`viewer_overreach`, the viewer's prefetch arithmetic): none may
    wrap, and the empty boxes must answer 200 with no body; and PATCHes of
    the last plane of each axis of levels 2, 4 and 8 of the layer, read back.
- `expected_diffs.json` lists the differences that are intended, each with a
  reason and, under `expect`, the candidate's answer: any of `status`,
  `len`, `sha256`, `text` (or `text_prefix`, the start of the text) for a
  response, `sha256` for a file, or `"unchanged"` for a file that must still
  equal the dataset as built. An entry whose `expect` is missing or empty,
  or pins only the status of a 200 that has a body, fails the run unless it
  carries `"unpinned_reason"` saying why the answer cannot be pinned; the
  run lists those. `expect_baseline` pins the baseline's answer the same
  way where it is deterministic, e.g. a crash that shows the case reaches
  the production bug. Anything not listed fails the run, and so does any
  candidate CRASH or TIMEOUT (even where the baseline crashes the same way),
  a listed difference that did not occur (the candidate behaves like the
  baseline there again), and a listed difference where an answer is not the
  expected one.
- Each server's full log is saved to `<work>/<role>.log`, and s18's other
  servers' to `<work>/<role>-skeleton-<value>.log`.

`--candidate-env KEY=VALUE` (repeatable) starts the candidate with that
environment variable, e.g. `CHUNK_CACHE_LINES=4096` to check that a larger
chunk cache changes no answer. CI runs the harness a second and third time
with `WRITE_BATCH_CHUNKS=64` and `=5`: batched chunk writes must store the
same bytes and give the same answers, so the allow file is the same.

`health.py --image IMAGE --work DIR [--lock-ms 700] [--runs N] [--require-hook]`
checks `/health` on one image, without a baseline: 200 `ok lock_wait_ms=N`
at idle (also with `READ_ONLY=1`), leaving `/performance` and every file as
they were; `HEALTH_LOCK_MS` taken, or ignored with one log line; and, while
a write is held inside the chunk cache lock, 503 `stuck lock_wait_ms=N
write_held_ms=M` within `HEALTH_LOCK_MS` + 0.5 s with no request thread left
waiting, then 200 again once the write goes through. Holding the write
needs a build made with `-DNTRACER_TEST_HOOKS=ON` (`docker build --build-arg
TEST_HOOKS=ON`; a write then waits while `/data/.test_hold_write_lock`
exists); on any other build that part is reported as SKIPPED, and
`--require-hook` makes that a failure.

`skip_unchanged.py --image IMAGE --work DIR [--rounds 40] [--require-hook] [--batch 0,64,3]`
checks `SKIP_UNCHANGED_CHUNKS=1` on one image, without a baseline, against a
server of the same image with the switch off: a PATCH of exactly what a layer
holds appends nothing (off: every chunk covered whole is appended again); one
changed voxel appends one frame and moves one entry; a box covering chunks in
part merges as before; zeros into an mchunk with no `.data` still answer 500,
a chunk with a corrupt frame is still replaced (in a whole-layer PATCH, only
that chunk is written), a video-compressed mchunk is still written; s19 with half the clients resending the same chunks; the
`/performance` counter line only with the switch on, and the "0" and invalid
settings. With a build made with `-DNTRACER_TEST_HOOKS=ON`, a PATCH waits
before its skip check while `/data/.test_hold_skip_check` exists, and the
script lands another PATCH into the same chunk, or replaces the mchunk's
files, in between: the held PATCH must still land, or answer 500 for the
replaced files, as without the switch; and `/health` must report a skip check
held inside the chunk cache lock (`/data/.test_hold_write_lock`) as a write.
Without the hook those cases are SKIPPED, and `--require-hook` makes that a
failure. `--batch 0,64,3` runs every case once per `WRITE_BATCH_CHUNKS`
value, 0 (not batched) first, each pass on fresh data and its own servers,
and then compares every stored file of the later passes with the first pass
byte for byte (except the s19 run's, whose frame order depends on timing, and
the hook cases' layers, written only in the later passes). In passes above 0
the `/performance` batch counters must move as the cases predict, the
start-up line must show in both logs, and four hook cases run on the server
with the skip switch off, on a 2x2x2-chunk box whose old voxels were read
into the cache first: an entry write that fails for one chunk
(`/data/.test_fail_entry_write` holding the chunk number), a `.meta` close
that fails (`/data/.test_fail_meta_close`), the mchunk's files replaced by
another version of the layer while a batch waits after its file check
(`/data/.test_hold_in_batch`; `/health` must report the held batch as a
write), and an append that fails (`/data/.test_fail_append`). Each must
answer 500, and every chunk must then read back as its files decode: the
cache keeps no old voxels of a chunk whose entry reached the file, after a
replacement every chunk of the layer reads the new files, which the write
left untouched, and a failed append moves no entry. The first pass also
checks the `WRITE_BATCH_CHUNKS` settings "0", "abc", "65"
and "64". CI runs it with `--batch 0,64,3` after the harness, on the release
build and on the sanitizer build, which has the hooks and adds
`--require-hook`. `concurrent_patch.py` takes `--env KEY=VALUE` and
`--resend-half` for the same s19 run on its own.

`stress_d6.py` is not part of the harness; CI runs it after it (below). It stresses stale chunks in the
chunk cache, which sit between two points inside the server and cannot be
triggered on demand from outside. `--mode patch`: a read that decoded a
chunk while a PATCH replaced it putting the old chunk back into the cache;
reader processes read one chunk without pause while a writer PATCHes it,
and after each PATCH the chunk is read again. `--box-edge 128` makes the box
128^3 (64 chunks, every one checked). `--slow-batch` (a build with the test
hooks, `WRITE_BATCH_CHUNKS` set) makes each batch wait 5 ms after its append
and before its entries (`/data/.test_slow_batch`), while reads go on without
reloading the unchanged `.meta`: a batch that bumped `data_gen` before its
entries then lets reads put old chunks back. `--fail-meta-close` (the same
build and setting) makes every batch's `.meta` close fail
(`/data/.test_fail_meta_close`): every PATCH must answer 500, and every chunk
read after it must still be the new one, since its entries reached the file
and the batch must still bump `data_gen`. `--mode rename`: s15's
sequence, repeated, reporting how many cycles landed. `--mode
rename-write`: the files replaced at a later point in each cycle of a long
PATCH, then every voxel checked; a PATCH that answered 500 must have left the
new files untouched. `--sweep auto` times one PATCH first and spreads the
renames over that time; `--require-mid-write` fails the run when no rename
landed during a PATCH's write-back. `--env KEY=VALUE` starts the server with
that variable; with `WRITE_BATCH_CHUNKS` set, `/performance` must show the
batch counters, and without it must not. CI runs `--mode patch` (300
PATCHes of one chunk), then with `WRITE_BATCH_CHUNKS=64`: `--mode patch
--box-edge 128` (with `--slow-batch` on the sanitizer build, which has the
hooks), `--mode patch --box-edge 128 --fail-meta-close` (sanitizer build
only), and `--mode rename-write --cycles 12 --sweep auto
--require-mid-write`. A run that finds no stale read means
something only if the same settings find some on a build without the fix.

Compare like with like: run both images on amd64. An integer division by zero
that kills the process on x86-64 returns 0 on arm64, so an arm64 candidate can
pass a case that crashes in production.
