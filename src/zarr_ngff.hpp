/*

SISF CDN - Scalable Image Stack Format Content Delivery Network
Copyright 2023-2026 by the Regents of the University of Michigan
Licensed under the terms specified in LICENSE.md

Zarr v3 / OME-NGFF metadata helpers.

This header has no TensorStore dependency, so the metadata parsing, the
mapping from Neuroglancer scale keys to pyramid levels, and the copy into the
SISF output byte order can be unit tested on their own
(see tests/zarr_ngff_test.cpp).

Supported layouts
-----------------
1. OME-NGFF multiscale group (OME-NGFF 0.5 on Zarr v3, "attributes.ome.multiscales",
   or the older "attributes.multiscales"). Every entry of datasets[] is one
   pyramid level and is a Zarr v3 array. Axis order comes from the NGFF "axes"
   (falling back to the array "dimension_names"), per-axis voxel size from the
   level's "scale" coordinate transformation. The Neuroglancer scale key of
   level i is attributes.<ns>.level_keys[i] when present, else 2**i, so the keys
   "1", "2", "4" keep the meaning they have for SISF datasets. Other members of
   the group (sub-groups, side files) are ignored.

2. A plain Zarr v3 array at the dataset root (the original ZARR dataset type).
   It is served as a single level with key 1. Axis order comes from
   "dimension_names" when present, otherwise x, y, z, c by position as before.

*/

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#else
#include "json.hpp"
#endif

namespace zarr_ngff
{
    using njson = nlohmann::json;

    // Attribute namespaces that may carry "level_keys" next to the NGFF metadata.
    static const char *const LEVEL_KEY_NAMESPACES[] = {"stitchreg", "sisf"};

    // ------------------------------------------------------------------
    // Data types
    // ------------------------------------------------------------------

    // The CDN serves 16 bit samples (the block route, filters, projections and
    // tracing all work on uint16). uint8 arrays are widened to uint16 on read,
    // so /info keeps "data_type": "uint16" for them. Any other Zarr data type
    // is rejected when the dataset is opened, with a message naming the type.
    enum class sample_type
    {
        unsupported,
        uint8,
        uint16
    };

    inline sample_type parse_sample_type(const std::string &zarr_data_type)
    {
        if (zarr_data_type == "uint16")
            return sample_type::uint16;
        if (zarr_data_type == "uint8")
            return sample_type::uint8;
        return sample_type::unsupported;
    }

    inline const char *sample_type_name(sample_type t)
    {
        switch (t)
        {
        case sample_type::uint8:
            return "uint8";
        case sample_type::uint16:
            return "uint16";
        default:
            return "unsupported";
        }
    }

    // ------------------------------------------------------------------
    // Zarr v3 array metadata (the array's zarr.json)
    // ------------------------------------------------------------------

    struct array_meta
    {
        std::vector<int64_t> shape;
        std::vector<std::string> dimension_names; // "" for unnamed dimensions, empty when absent
        std::string data_type;
        std::vector<int64_t> chunk_shape;       // regular grid chunk (the shard when sharded)
        std::vector<int64_t> inner_chunk_shape; // sharding_indexed inner chunk, else chunk_shape
        bool sharded = false;
        std::vector<std::string> codecs; // codec names, outer chain first, then the inner (sharded) chain
    };

    inline void collect_codecs(const njson &chain, array_meta &out, std::vector<int64_t> *inner_shape)
    {
        if (!chain.is_array())
            return;
        for (const auto &c : chain)
        {
            if (!c.is_object() || !c.contains("name") || !c["name"].is_string())
                continue;
            const std::string name = c["name"].get<std::string>();
            out.codecs.push_back(name);
            if (name == "sharding_indexed" && c.contains("configuration"))
            {
                const njson &cfg = c["configuration"];
                out.sharded = true;
                if (inner_shape != nullptr && cfg.contains("chunk_shape") && cfg["chunk_shape"].is_array())
                {
                    inner_shape->clear();
                    for (const auto &v : cfg["chunk_shape"])
                        inner_shape->push_back(v.get<int64_t>());
                }
                if (cfg.contains("codecs"))
                    collect_codecs(cfg["codecs"], out, nullptr);
            }
        }
    }

    inline bool parse_array_meta(const njson &meta, array_meta &out, std::string &err)
    {
        out = array_meta();
        if (!meta.is_object())
        {
            err = "array metadata is not a JSON object";
            return false;
        }
        if (meta.contains("zarr_format") && meta["zarr_format"] != 3)
        {
            err = "only zarr_format 3 is supported";
            return false;
        }
        if (!meta.contains("node_type") || meta["node_type"] != "array")
        {
            err = "node_type is not \"array\"";
            return false;
        }
        if (!meta.contains("shape") || !meta["shape"].is_array())
        {
            err = "array has no shape";
            return false;
        }
        for (const auto &v : meta["shape"])
            out.shape.push_back(v.get<int64_t>());

        if (meta.contains("data_type") && meta["data_type"].is_string())
            out.data_type = meta["data_type"].get<std::string>();
        else
        {
            err = "array has no (string) data_type";
            return false;
        }

        if (meta.contains("dimension_names") && meta["dimension_names"].is_array())
        {
            for (const auto &v : meta["dimension_names"])
                out.dimension_names.push_back(v.is_string() ? v.get<std::string>() : std::string());
            if (out.dimension_names.size() != out.shape.size())
            {
                err = "dimension_names length does not match the array rank";
                return false;
            }
        }

        if (meta.contains("chunk_grid") && meta["chunk_grid"].is_object())
        {
            const njson &grid = meta["chunk_grid"];
            if (grid.contains("configuration") && grid["configuration"].contains("chunk_shape"))
            {
                for (const auto &v : grid["configuration"]["chunk_shape"])
                    out.chunk_shape.push_back(v.get<int64_t>());
            }
        }
        if (out.chunk_shape.size() != out.shape.size())
            out.chunk_shape = out.shape;

        std::vector<int64_t> inner;
        if (meta.contains("codecs"))
            collect_codecs(meta["codecs"], out, &inner);
        out.inner_chunk_shape = inner.size() == out.shape.size() ? inner : out.chunk_shape;
        return true;
    }

    // ------------------------------------------------------------------
    // Axis order
    // ------------------------------------------------------------------

    // Position of each spatial / channel axis in the array's dimension order.
    // "other" lists any further axes (for example t); they are read at index 0.
    struct axis_map
    {
        int x = -1, y = -1, z = -1, c = -1;
        std::vector<int> other;
        bool positional = false; // true when no axis names were available
    };

    inline std::string lower(std::string s)
    {
        for (auto &ch : s)
            ch = (char)std::tolower((unsigned char)ch);
        return s;
    }

    // names: one entry per array dimension ("" when unnamed), or empty.
    inline bool resolve_axes(const std::vector<std::string> &names, size_t rank, axis_map &out, std::string &err)
    {
        out = axis_map();

        bool any_named = false;
        for (const auto &n : names)
            any_named |= !n.empty();

        if (names.empty() || !any_named)
        {
            // Legacy plain array without names: x, y, z, c by position.
            if (rank != 3 && rank != 4)
            {
                err = "unnamed axes need rank 3 (x,y,z) or 4 (x,y,z,c)";
                return false;
            }
            out.positional = true;
            out.x = 0;
            out.y = 1;
            out.z = 2;
            out.c = rank == 4 ? 3 : -1;
            return true;
        }

        if (names.size() != rank)
        {
            err = "number of axis names does not match the array rank";
            return false;
        }

        for (size_t i = 0; i < rank; i++)
        {
            const std::string n = lower(names[i]);
            int *slot = nullptr;
            if (n == "x")
                slot = &out.x;
            else if (n == "y")
                slot = &out.y;
            else if (n == "z")
                slot = &out.z;
            else if (n == "c" || n == "ch" || n == "channel" || n == "channels")
                slot = &out.c;

            if (slot == nullptr)
            {
                out.other.push_back((int)i);
                continue;
            }
            if (*slot != -1)
            {
                err = "axis \"" + names[i] + "\" appears twice";
                return false;
            }
            *slot = (int)i;
        }

        if (out.x < 0 || out.y < 0 || out.z < 0)
        {
            err = "axes must include x, y and z";
            return false;
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Units
    // ------------------------------------------------------------------

    // Nanometres per unit for the NGFF / UDUNITS-2 space units. An empty unit
    // is taken as nanometres (the CDN's native unit).
    inline bool unit_to_nm(const std::string &unit_in, double &factor)
    {
        const std::string u = lower(unit_in);
        if (u.empty() || u == "nanometer" || u == "nanometre" || u == "nm")
            factor = 1.0;
        else if (u == "micrometer" || u == "micrometre" || u == "micron" || u == "um" ||
                 u == "\u00b5m" || u == "\u03bcm")
            factor = 1e3;
        else if (u == "millimeter" || u == "millimetre" || u == "mm")
            factor = 1e6;
        else if (u == "centimeter" || u == "centimetre" || u == "cm")
            factor = 1e7;
        else if (u == "meter" || u == "metre" || u == "m")
            factor = 1e9;
        else if (u == "angstrom")
            factor = 0.1;
        else if (u == "picometer" || u == "picometre" || u == "pm")
            factor = 1e-3;
        else
            return false;
        return true;
    }

    // ------------------------------------------------------------------
    // OME-NGFF multiscale group
    // ------------------------------------------------------------------

    struct multiscale_level
    {
        std::string path;
        size_t key = 0;            // Neuroglancer scale key
        std::vector<double> scale; // per NGFF axis, NGFF units (product with any global scale)
    };

    struct multiscale
    {
        std::string name;
        std::vector<std::string> axis_names; // NGFF axes, in array dimension order
        std::vector<std::string> axis_units;
        std::vector<multiscale_level> levels;
        bool keys_from_metadata = false;
    };

    inline bool read_scale_transform(const njson &transforms, std::vector<double> &scale)
    {
        if (!transforms.is_array())
            return false;
        for (const auto &t : transforms)
        {
            if (t.is_object() && t.contains("type") && t["type"] == "scale" && t.contains("scale") &&
                t["scale"].is_array())
            {
                scale.clear();
                for (const auto &v : t["scale"])
                    scale.push_back(v.get<double>());
                return true;
            }
        }
        return false;
    }

    inline bool parse_level_key(const njson &v, size_t &key)
    {
        if (v.is_number_unsigned() || v.is_number_integer())
        {
            const int64_t k = v.get<int64_t>();
            if (k <= 0)
                return false;
            key = (size_t)k;
            return true;
        }
        if (v.is_string())
        {
            const std::string s = v.get<std::string>();
            if (s.empty() || s.size() > 18)
                return false;
            for (char ch : s)
                if (!std::isdigit((unsigned char)ch))
                    return false;
            const size_t k = (size_t)std::stoull(s);
            if (k == 0)
                return false;
            key = k;
            return true;
        }
        return false;
    }

    // True when the group metadata describes an NGFF multiscale image.
    inline bool find_multiscales(const njson &group_meta, const njson *&ms_out)
    {
        ms_out = nullptr;
        if (!group_meta.is_object() || !group_meta.contains("attributes"))
            return false;
        const njson &attrs = group_meta["attributes"];
        if (attrs.contains("ome") && attrs["ome"].is_object() && attrs["ome"].contains("multiscales") &&
            attrs["ome"]["multiscales"].is_array() && !attrs["ome"]["multiscales"].empty())
        {
            ms_out = &attrs["ome"]["multiscales"][0];
            return true;
        }
        if (attrs.contains("multiscales") && attrs["multiscales"].is_array() && !attrs["multiscales"].empty())
        {
            ms_out = &attrs["multiscales"][0];
            return true;
        }
        return false;
    }

    inline bool parse_multiscale_group(const njson &group_meta, multiscale &out, std::string &err)
    {
        out = multiscale();

        if (!group_meta.is_object() || !group_meta.contains("node_type") || group_meta["node_type"] != "group")
        {
            err = "node_type is not \"group\"";
            return false;
        }

        const njson *ms = nullptr;
        if (!find_multiscales(group_meta, ms))
        {
            err = "group has no OME-NGFF multiscales metadata";
            return false;
        }

        if (ms->contains("name") && (*ms)["name"].is_string())
            out.name = (*ms)["name"].get<std::string>();

        // Axes: objects {name, type, unit} (NGFF >= 0.4) or plain strings (0.3).
        if (ms->contains("axes") && (*ms)["axes"].is_array())
        {
            for (const auto &a : (*ms)["axes"])
            {
                if (a.is_string())
                {
                    out.axis_names.push_back(a.get<std::string>());
                    out.axis_units.push_back("");
                }
                else if (a.is_object())
                {
                    out.axis_names.push_back(a.contains("name") && a["name"].is_string() ? a["name"].get<std::string>() : "");
                    out.axis_units.push_back(a.contains("unit") && a["unit"].is_string() ? a["unit"].get<std::string>() : "");
                }
            }
        }

        // Optional multiscale-wide scale, applied on top of every level's scale.
        std::vector<double> global_scale;
        if (ms->contains("coordinateTransformations"))
            read_scale_transform((*ms)["coordinateTransformations"], global_scale);

        if (!ms->contains("datasets") || !(*ms)["datasets"].is_array() || (*ms)["datasets"].empty())
        {
            err = "multiscales[0] has no datasets";
            return false;
        }

        for (const auto &d : (*ms)["datasets"])
        {
            multiscale_level lvl;
            if (!d.is_object() || !d.contains("path") || !d["path"].is_string())
            {
                err = "multiscales dataset without a path";
                return false;
            }
            lvl.path = d["path"].get<std::string>();
            if (lvl.path.empty() || lvl.path.find("..") != std::string::npos || lvl.path[0] == '/')
            {
                err = "invalid dataset path \"" + lvl.path + "\"";
                return false;
            }

            if (d.contains("coordinateTransformations"))
                read_scale_transform(d["coordinateTransformations"], lvl.scale);
            if (!global_scale.empty() && global_scale.size() == lvl.scale.size())
                for (size_t i = 0; i < lvl.scale.size(); i++)
                    lvl.scale[i] *= global_scale[i];

            out.levels.push_back(lvl);
        }

        // Scale keys: <ns>.level_keys when present and valid, else 2**i.
        const njson &attrs = group_meta["attributes"];
        const njson *keys = nullptr;
        for (const char *ns : LEVEL_KEY_NAMESPACES)
        {
            if (attrs.contains(ns) && attrs[ns].is_object() && attrs[ns].contains("level_keys") &&
                attrs[ns]["level_keys"].is_array())
            {
                keys = &attrs[ns]["level_keys"];
                break;
            }
        }

        bool keys_ok = false;
        if (keys != nullptr && keys->size() == out.levels.size())
        {
            keys_ok = true;
            std::set<size_t> seen;
            for (size_t i = 0; i < out.levels.size(); i++)
            {
                size_t k = 0;
                if (!parse_level_key((*keys)[i], k) || seen.count(k) != 0)
                {
                    keys_ok = false;
                    break;
                }
                seen.insert(k);
                out.levels[i].key = k;
            }
        }
        out.keys_from_metadata = keys_ok;
        if (!keys_ok)
        {
            for (size_t i = 0; i < out.levels.size(); i++)
                out.levels[i].key = ((size_t)1) << i;
        }

        return true;
    }

    // ------------------------------------------------------------------
    // Served view of one level
    // ------------------------------------------------------------------

    struct level_view
    {
        std::string path; // relative to the dataset root, "" for a root array
        size_t key = 1;   // Neuroglancer scale key
        array_meta array;
        axis_map axes;
        sample_type dtype = sample_type::unsupported;

        size_t size_x = 0, size_y = 0, size_z = 0, channels = 1;
        double res_nm_x = 0, res_nm_y = 0, res_nm_z = 0;

        // Chunk shapes in x, y, z (shard and inner chunk)
        size_t chunk_x = 0, chunk_y = 0, chunk_z = 0;
        size_t inner_x = 0, inner_y = 0, inner_z = 0;
    };

    // Fill sizes, channel count, dtype and chunk shapes from the array metadata
    // and an already resolved axis map. Resolution is left to the caller.
    inline bool fill_level_view(level_view &v, std::string &err)
    {
        const array_meta &a = v.array;
        const axis_map &m = v.axes;
        const int rank = (int)a.shape.size();
        auto in_range = [rank](int d) { return d >= 0 && d < rank; };
        if (!in_range(m.x) || !in_range(m.y) || !in_range(m.z) || (m.c != -1 && !in_range(m.c)))
        {
            err = "axis index out of range for the array rank";
            return false;
        }

        v.dtype = parse_sample_type(a.data_type);
        if (v.dtype == sample_type::unsupported)
        {
            err = "unsupported data_type \"" + a.data_type + "\" (served types: uint16, uint8)";
            return false;
        }

        v.size_x = (size_t)a.shape[m.x];
        v.size_y = (size_t)a.shape[m.y];
        v.size_z = (size_t)a.shape[m.z];
        v.channels = m.c >= 0 ? (size_t)a.shape[m.c] : 1;
        if (v.size_x == 0 || v.size_y == 0 || v.size_z == 0 || v.channels == 0)
        {
            err = "array has an empty dimension";
            return false;
        }

        v.chunk_x = (size_t)a.chunk_shape[m.x];
        v.chunk_y = (size_t)a.chunk_shape[m.y];
        v.chunk_z = (size_t)a.chunk_shape[m.z];
        v.inner_x = (size_t)a.inner_chunk_shape[m.x];
        v.inner_y = (size_t)a.inner_chunk_shape[m.y];
        v.inner_z = (size_t)a.inner_chunk_shape[m.z];
        return true;
    }

    // Build the served view of every level of an NGFF multiscale group.
    // arrays[i] is the parsed zarr.json of ms.levels[i].
    inline bool build_multiscale_views(const multiscale &ms, const std::vector<array_meta> &arrays,
                                       std::vector<level_view> &out, std::string &err)
    {
        out.clear();
        if (arrays.size() != ms.levels.size())
        {
            err = "level count mismatch";
            return false;
        }

        for (size_t i = 0; i < ms.levels.size(); i++)
        {
            level_view v;
            v.path = ms.levels[i].path;
            v.key = ms.levels[i].key;
            v.array = arrays[i];
            const size_t rank = v.array.shape.size();

            // Axis names: NGFF axes first, array dimension_names second.
            const std::vector<std::string> &names = !ms.axis_names.empty() ? ms.axis_names : v.array.dimension_names;
            if (names.empty())
            {
                err = "level \"" + v.path + "\": no axis names in the NGFF axes or the array dimension_names";
                return false;
            }
            if (!resolve_axes(names, rank, v.axes, err))
            {
                err = "level \"" + v.path + "\": " + err;
                return false;
            }
            if (!fill_level_view(v, err))
            {
                err = "level \"" + v.path + "\": " + err;
                return false;
            }

            // Per-axis voxel size in nm from the level's scale transform.
            const std::vector<double> &s = ms.levels[i].scale;
            if (s.size() != rank)
            {
                err = "level \"" + v.path + "\": scale transform missing or of the wrong length";
                return false;
            }
            const int spatial[3] = {v.axes.x, v.axes.y, v.axes.z};
            double res[3];
            for (int k = 0; k < 3; k++)
            {
                const int d = spatial[k];
                double f = 1.0;
                const std::string unit = (size_t)d < ms.axis_units.size() ? ms.axis_units[d] : "";
                if (!unit_to_nm(unit, f))
                {
                    err = "level \"" + v.path + "\": unsupported space unit \"" + unit + "\"";
                    return false;
                }
                res[k] = s[d] * f;
                if (!(res[k] > 0) || !std::isfinite(res[k]))
                {
                    err = "level \"" + v.path + "\": non-positive voxel size";
                    return false;
                }
            }
            v.res_nm_x = res[0];
            v.res_nm_y = res[1];
            v.res_nm_z = res[2];

            if (!out.empty())
            {
                if (v.channels != out[0].channels)
                {
                    err = "level \"" + v.path + "\": channel count differs from level 0";
                    return false;
                }
                if (v.dtype != out[0].dtype)
                {
                    err = "level \"" + v.path + "\": data_type differs from level 0";
                    return false;
                }
            }
            out.push_back(v);
        }
        return true;
    }

    inline int find_level(const std::vector<level_view> &levels, size_t key)
    {
        for (size_t i = 0; i < levels.size(); i++)
            if (levels[i].key == key)
                return (int)i;
        return -1;
    }

    // ------------------------------------------------------------------
    // Output byte order
    // ------------------------------------------------------------------

    // Copy a region read in the array's own dimension order into the SISF
    // block layout: uint16, channel slowest, then z, then y, x fastest
    // (Neuroglancer "raw" encoding, Fortran order over [x, y, z, channel]).
    //
    //   out[((c * nz + z) * ny + y) * nx + x]
    //
    // src points at element (0, ..., 0) of the region; byte_strides has one
    // entry per array dimension. Dimensions not in the axis map (for example t)
    // must have extent 1 in the region and are read at index 0.
    template <typename T>
    inline void copy_region_to_czyx_u16(const T *src, const std::vector<int64_t> &byte_strides, const axis_map &axes,
                                        size_t nc, size_t nz, size_t ny, size_t nx, uint16_t *out)
    {
        const char *base = reinterpret_cast<const char *>(src);
        const int64_t sx = byte_strides[axes.x];
        const int64_t sy = byte_strides[axes.y];
        const int64_t sz = byte_strides[axes.z];
        const int64_t sc = axes.c >= 0 ? byte_strides[axes.c] : 0;

        size_t o = 0;
        for (size_t c = 0; c < nc; c++)
            for (size_t z = 0; z < nz; z++)
                for (size_t y = 0; y < ny; y++)
                {
                    const char *row = base + (int64_t)c * sc + (int64_t)z * sz + (int64_t)y * sy;
                    if (sizeof(T) == sizeof(uint16_t) && sx == (int64_t)sizeof(T))
                    {
                        std::memcpy(out + o, row, nx * sizeof(uint16_t));
                        o += nx;
                    }
                    else
                    {
                        for (size_t x = 0; x < nx; x++)
                        {
                            T v;
                            std::memcpy(&v, row + (int64_t)x * sx, sizeof(T));
                            out[o++] = (uint16_t)v;
                        }
                    }
                }
    }
}
