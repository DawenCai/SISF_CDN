// Persisted flat-field planes: Zarr v3 arrays [2, h, w] (flat, dark), float16, beside the model
// (see flatfield.hpp). Installs FlatfieldModel's load/save hooks at startup. Reads go through a
// TensorStore cast to float32, so arrays written as float32 or uint16 by other tools load too.
#pragma once

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "flatfield.hpp"
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/tensorstore.h"

namespace flatfield_store
{
inline nlohmann::json cast_spec(const std::string &path)
{
    return {{"driver", "cast"},
            {"dtype", "float32"},
            {"base", {{"driver", "zarr3"}, {"kvstore", {{"driver", "file"}, {"path", path}}}}}};
}

inline bool load(const std::string &path, FlatfieldModel::Plane &pl)
{
    std::error_code ec;
    if (!std::filesystem::exists(path + "/zarr.json", ec))
        return false;
    auto store = tensorstore::Open(cast_spec(path), tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read).result();
    if (!store.ok())
        return false;
    auto shape = store->domain().shape();
    if (shape.size() != 3 || shape[0] != 2 || (size_t)shape[1] != pl.h || (size_t)shape[2] != pl.w)
        return false;
    auto arr = tensorstore::Read<tensorstore::zero_origin>(*store).result();
    if (!arr.ok())
        return false;
    const float *v = static_cast<const float *>(arr->data());
    const size_t n = pl.h * pl.w;
    pl.inv_flat.resize(n);
    pl.dark.assign(v + n, v + 2 * n);
    for (size_t i = 0; i < n; i++)
        pl.inv_flat[i] = 1.0f / std::max(v[i], 1e-3f);
    return true;
}

inline bool save(const std::string &path, FlatfieldModel::Plane &pl)
{
    try
    {
        const size_t n = pl.h * pl.w;
        std::vector<float> buf(2 * n);
        for (size_t i = 0; i < n; i++)
            buf[i] = 1.0f / pl.inv_flat[i];
        std::copy(pl.dark.begin(), pl.dark.end(), buf.begin() + n);
        const std::string tmp = path + ".part";
        std::error_code ec;
        std::filesystem::remove_all(tmp, ec);
        nlohmann::json spec = cast_spec(tmp);
        spec["base"]["create"] = true;
        spec["base"]["metadata"] = {
            {"shape", {2, pl.h, pl.w}},
            {"data_type", "float16"},
            {"chunk_grid", {{"name", "regular"}, {"configuration", {{"chunk_shape", {2, pl.h, pl.w}}}}}},
            {"codecs", {{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}, {{"name", "zstd"}, {"configuration", {{"level", 3}}}}}},
        };
        auto store = tensorstore::Open(spec, tensorstore::OpenMode::create, tensorstore::ReadWriteMode::write).result();
        if (!store.ok())
            return false; // read-only data folder: keep the plane in memory only
        auto src = tensorstore::AllocateArray<float>({2, (tensorstore::Index)pl.h, (tensorstore::Index)pl.w});
        std::copy(buf.begin(), buf.end(), src.data());
        if (!tensorstore::Write(src, *store).commit_future.result().ok())
            return false;
        std::filesystem::rename(tmp, path, ec); // readers never see a half-written array
        return !ec;
    }
    catch (const std::exception &e)
    {
        return false;
    }
}

struct Installer
{
    Installer()
    {
        FlatfieldModel::load_hook() = load;
        FlatfieldModel::save_hook() = save;
    }
};
inline Installer installer;
} // namespace flatfield_store
