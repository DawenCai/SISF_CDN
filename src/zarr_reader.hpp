/*

SISF CDN - Scalable Image Stack Format Content Delivery Network
Copyright 2023-2026 by the Regents of the University of Michigan
Licensed under the terms specified in LICENSE.md

Zarr v3 / OME-NGFF dataset reader backed by TensorStore.

Caching
-------
All Zarr datasets share one process-wide tensorstore::Context (zarr_context()).
Its cache_pool has a bounded total_bytes_limit (default 1 GiB, environment
variable ZARR_CACHE_POOL_BYTES overrides it; 0 disables caching). Every level
array is opened once, on first use, and the open TensorStore is kept for the
life of the process, so:

  * decoded inner chunks are cached in the pool and reused across requests;
  * for sharded arrays, the zarr3 sharding_indexed driver caches each shard's
    index in the same pool, keyed by the shard file. A block request therefore
    reads the shard index once, not once per inner chunk or per request. The
    cache is shared by all levels of a dataset (and by all datasets), and the
    limit above is the total for the process, not per dataset.

Context::Default() has a cache_pool limit of 0, which is why the earlier
per-request open had neither a chunk cache nor a shard-index cache.

TensorStore revalidates cached entries against the file generation on each
read (recheck_cached_data defaults to true), so a rewritten shard is picked up
without a restart, as SISF does when its .meta files change.

Custom decoders
---------------
TensorStore decodes the array's codec chain (bytes, zstd, blosc, sharding...).
An array whose chunks need a decoder TensorStore does not have (for example a
video codec stored inside the shards) can be served by registering a factory
in zarr_custom_level_sources(). Factories are asked in order when a level is
first opened; the first one that returns a source serves that level, otherwise
the level is read through TensorStore. Nothing is registered by default.

*/

#pragma once

#include <stdint.h>
#include <stdlib.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/unit.h"

#include "zarr_ngff.hpp"

// ----------------------------------------------------------------------
// Shared context
// ----------------------------------------------------------------------

const int64_t ZARR_DEFAULT_CACHE_POOL_BYTES = ((int64_t)1) << 30; // 1 GiB

inline int64_t zarr_cache_pool_bytes()
{
    const char *env = std::getenv("ZARR_CACHE_POOL_BYTES");
    if (env != nullptr && env[0] != '\0')
    {
        char *end = nullptr;
        const long long v = std::strtoll(env, &end, 10);
        if (end != env && v >= 0)
            return (int64_t)v;
        std::cerr << "Ignoring invalid ZARR_CACHE_POOL_BYTES=\"" << env << "\"" << std::endl;
    }
    return ZARR_DEFAULT_CACHE_POOL_BYTES;
}

inline tensorstore::Context zarr_context()
{
    static std::once_flag once;
    static tensorstore::Context context;

    std::call_once(once, []()
                   {
        const int64_t bytes = zarr_cache_pool_bytes();
        auto ctx = tensorstore::Context::FromJson({{"cache_pool", {{"total_bytes_limit", bytes}}}});
        if (ctx.ok())
        {
            context = *ctx;
            std::cout << "Zarr cache pool: " << bytes << " bytes" << std::endl;
        }
        else
        {
            std::cerr << "Zarr context spec rejected (" << ctx.status() << "), using the default context (no cache)" << std::endl;
            context = tensorstore::Context::Default();
        } });

    return context;
}

// ----------------------------------------------------------------------
// Level sources
// ----------------------------------------------------------------------

// Reads one region of one level into the SISF block layout
// (uint16, channel slowest, x fastest; see zarr_ngff::copy_region_to_czyx_u16).
class zarr_level_source
{
public:
    virtual ~zarr_level_source() {}

    // [xs, xe) x [ys, ye) x [zs, ze), all channels. out holds
    // channels * (ze - zs) * (ye - ys) * (xe - xs) uint16 values.
    virtual bool read(size_t xs, size_t xe, size_t ys, size_t ye, size_t zs, size_t ze,
                      uint16_t *out, std::string &err) = 0;
};

// Hook for a per-array custom decoder. Return nullptr to decline the level.
// array_path is the level array's directory, level its parsed metadata
// (level.array.codecs lists the codec names, outer chain first).
typedef std::function<std::unique_ptr<zarr_level_source>(const std::string &array_path,
                                                          const zarr_ngff::level_view &level)>
    zarr_level_source_factory;

inline std::vector<zarr_level_source_factory> &zarr_custom_level_sources()
{
    static std::vector<zarr_level_source_factory> factories;
    return factories;
}

inline tensorstore::Result<tensorstore::TensorStore<>> zarr_open_array(const std::string &array_path)
{
    nlohmann::json spec = {
        {"driver", "zarr3"},
        {"kvstore", {{"driver", "file"}, {"path", array_path}}},
    };

    return tensorstore::Open(spec, zarr_context(), tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read).result();
}

class tensorstore_level_source : public zarr_level_source
{
public:
    tensorstore::TensorStore<> store;
    zarr_ngff::level_view level;

    tensorstore_level_source(tensorstore::TensorStore<> store_in, const zarr_ngff::level_view &level_in)
        : store(std::move(store_in)), level(level_in) {}

    bool read(size_t xs, size_t xe, size_t ys, size_t ye, size_t zs, size_t ze,
              uint16_t *out, std::string &err) override
    {
        const zarr_ngff::axis_map &axes = level.axes;
        const size_t rank = (size_t)store.rank();
        if (rank != level.array.shape.size())
        {
            err = "rank changed since the dataset was opened";
            return false;
        }

        const size_t nx = xe - xs, ny = ye - ys, nz = ze - zs, nc = level.channels;

        // Region in the array's own dimension order. Extra axes (t, ...) are
        // read at their origin with extent 1.
        std::vector<tensorstore::Index> start(rank), size(rank);
        auto origin = store.domain().origin();
        for (size_t d = 0; d < rank; d++)
        {
            start[d] = origin[d];
            size[d] = 1;
        }
        start[axes.x] += (tensorstore::Index)xs;
        size[axes.x] = (tensorstore::Index)nx;
        start[axes.y] += (tensorstore::Index)ys;
        size[axes.y] = (tensorstore::Index)ny;
        start[axes.z] += (tensorstore::Index)zs;
        size[axes.z] = (tensorstore::Index)nz;
        if (axes.c >= 0)
            size[axes.c] = (tensorstore::Index)nc;

        auto view = store | tensorstore::AllDims().SizedInterval(
                                tensorstore::span<const tensorstore::Index>(start.data(), rank),
                                tensorstore::span<const tensorstore::Index>(size.data(), rank));
        if (!view.ok())
        {
            err = std::string("invalid region: ") + view.status().ToString();
            return false;
        }

        auto array_result = tensorstore::Read<tensorstore::zero_origin>(*view).result();
        if (!array_result.ok())
        {
            err = std::string("read failed: ") + array_result.status().ToString();
            return false;
        }
        const auto &array = *array_result;

        std::vector<int64_t> byte_strides(rank);
        for (size_t d = 0; d < rank; d++)
            byte_strides[d] = (int64_t)array.byte_strides()[d];

        switch (level.dtype)
        {
        case zarr_ngff::sample_type::uint16:
            if (array.dtype() != tensorstore::dtype_v<uint16_t>)
            {
                err = "array data type is not uint16";
                return false;
            }
            zarr_ngff::copy_region_to_czyx_u16(static_cast<const uint16_t *>(array.data()), byte_strides, axes,
                                               nc, nz, ny, nx, out);
            return true;

        case zarr_ngff::sample_type::uint8:
            // Widened to uint16; /info reports uint16 for every dataset.
            if (array.dtype() != tensorstore::dtype_v<uint8_t>)
            {
                err = "array data type is not uint8";
                return false;
            }
            zarr_ngff::copy_region_to_czyx_u16(static_cast<const uint8_t *>(array.data()), byte_strides, axes,
                                               nc, nz, ny, nx, out);
            return true;

        default:
            err = "unsupported data type";
            return false;
        }
    }
};

// ----------------------------------------------------------------------
// Dataset
// ----------------------------------------------------------------------

// Thrown by archive_reader::load_region when a Zarr read fails, so the
// request that hit it gets an HTTP error and the server keeps running.
class zarr_read_error : public std::runtime_error
{
public:
    explicit zarr_read_error(const std::string &msg) : std::runtime_error(msg) {}
};

class zarr_dataset
{
public:
    std::string root;
    bool multiscale = false; // false: plain array at the root
    std::vector<zarr_ngff::level_view> levels;

    // Opens the dataset metadata. Throws std::runtime_error with a one-line
    // reason when the dataset cannot be served (missing or malformed
    // zarr.json, unknown layout, unsupported dtype, a root array TensorStore
    // cannot open). Nothing here aborts the process; the inventory loader
    // catches the exception, logs it and skips only this dataset.
    explicit zarr_dataset(const std::string &root_in) : root(root_in)
    {
        zarr_ngff::root_kind kind;
        std::string err;
        if (!zarr_ngff::load_dataset_metadata(root, kind, levels, err))
            throw std::runtime_error(err);

        multiscale = kind == zarr_ngff::root_kind::multiscale;
        if (!multiscale)
            open_plain_array();

        sources.resize(levels.size());
        source_mutexes.reset(new std::mutex[levels.size()]);
        if (preopened)
            sources[0] = std::move(preopened);
    }

    const zarr_ngff::level_view *level_for_key(size_t key) const
    {
        const int i = zarr_ngff::find_level(levels, key);
        return i < 0 ? nullptr : &levels[(size_t)i];
    }

    // Read [xs, xe) x [ys, ye) x [zs, ze) of the level with scale key `key`
    // into out (SISF block layout). Returns false with a reason in err when
    // the level cannot be opened or read; out may then be partly written.
    bool read(size_t key, size_t xs, size_t xe, size_t ys, size_t ye, size_t zs, size_t ze, uint16_t *out,
              std::string &err)
    {
        const int i = zarr_ngff::find_level(levels, key);
        if (i < 0)
        {
            err = "no level for scale key " + std::to_string(key);
            return false;
        }

        zarr_level_source *src = get_source((size_t)i, err);
        if (src == nullptr)
            return false;

        try
        {
            if (!src->read(xs, xe, ys, ye, zs, ze, out, err))
            {
                err = "level \"" + levels[i].path + "\": " + err;
                return false;
            }
        }
        catch (const std::exception &e)
        {
            err = "level \"" + levels[i].path + "\": " + e.what();
            return false;
        }
        return true;
    }

private:
    std::vector<std::unique_ptr<zarr_level_source>> sources;
    std::unique_ptr<std::mutex[]> source_mutexes; // one per level, guards the lazy open
    std::unique_ptr<zarr_level_source> preopened;

    std::string level_dir(const zarr_ngff::level_view &v) const
    {
        return v.path.empty() ? root : root + "/" + v.path;
    }

    // The original ZARR dataset type: one array at the root, served as key 1.
    // Opened now (as before) so its resolution can come from TensorStore's
    // dimension units; the open store is kept and reused for every request.
    void open_plain_array()
    {
        zarr_ngff::level_view &v = levels[0];

        auto store = zarr_open_array(root);
        if (!store.ok())
            throw std::runtime_error(std::string("cannot open array: ") + store.status().ToString());

        // Resolution from dimension units when they are set, else 100 nm (as before).
        v.res_nm_x = v.res_nm_y = v.res_nm_z = 100;
        auto units = store->dimension_units();
        if (units.ok())
        {
            const int spatial[3] = {v.axes.x, v.axes.y, v.axes.z};
            double *res[3] = {&v.res_nm_x, &v.res_nm_y, &v.res_nm_z};
            for (int k = 0; k < 3; k++)
            {
                const size_t d = (size_t)spatial[k];
                if (d < units->size() && (*units)[d].has_value())
                {
                    const tensorstore::Unit &u = *(*units)[d];
                    double f = 1.0;
                    if (!zarr_ngff::unit_to_nm(u.base_unit, f))
                        f = 1.0; // unknown unit: keep the multiplier as nm, as before
                    if (u.multiplier * f > 0)
                        *res[k] = u.multiplier * f;
                }
            }
        }

        preopened.reset(new tensorstore_level_source(std::move(*store), v));
    }

    zarr_level_source *get_source(size_t i, std::string &err)
    {
        std::lock_guard<std::mutex> lock(source_mutexes[i]);
        if (sources[i])
            return sources[i].get();

        const zarr_ngff::level_view &v = levels[i];
        const std::string dir = level_dir(v);

        try
        {
            for (auto &factory : zarr_custom_level_sources())
            {
                std::unique_ptr<zarr_level_source> s = factory(dir, v);
                if (s)
                {
                    sources[i] = std::move(s);
                    return sources[i].get();
                }
            }
        }
        catch (const std::exception &e)
        {
            err = "custom decoder for level \"" + v.path + "\" failed: " + e.what();
            return nullptr;
        }

        // A failed open is not cached, so the next request retries it.
        auto store = zarr_open_array(dir);
        if (!store.ok())
        {
            err = "cannot open level \"" + v.path + "\": " + store.status().ToString();
            return nullptr;
        }
        if ((size_t)store->rank() != v.array.shape.size())
        {
            err = "level \"" + v.path + "\": rank differs from zarr.json";
            return nullptr;
        }

        sources[i].reset(new tensorstore_level_source(std::move(*store), v));
        return sources[i].get();
    }
};
