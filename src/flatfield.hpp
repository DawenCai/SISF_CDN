// Flat-field (BaSiC) correction applied while serving, as a filter: (raw - dark) / flat.
//
// The profiles are fitted by SAVAII's FLATFIELD_FIT job. To use them here they are exported next
// to the dataset as raw float32 files plus a small JSON index:
//
//   <dataset>/.flatfield/model.json
//   {
//     "use_darkfield": true,
//     "channels": {"0": {"flat": "c0_flat.f32", "dark": "c0_dark.f32", "shape": [h, w]}},
//     "tile_grid": {"cell": [x, y], "window_origin": [x, y], "fov": [x, y]}   (optional, Zarr)
//   }
//
// Profiles are row-major (y, x) float32 over the full camera field of view, as SAVAII stores
// them. A SISF shard holds the full field of view (overlaps are hidden by its crop window), so
// a voxel at raw shard position (x, y) uses the profile resized to the shard size at that
// pyramid level.
//
// The resize is done once per (channel, level size, model version), never per request: the
// result (1 / flat and dark, full resolution for that level) is kept in memory and, through the
// optional load/save hooks, persisted beside the model as
//   <dataset>/.flatfield/levels/c<channel>_<w>x<h>_<version>.zarr   (Zarr v3, float16, [2, h, w])
// so a restart does not recompute; SAVAII's export writes these arrays too. The model version is
// model.json's "version", else a hash of model.json. A request looks a plane up once per shard
// (or level) and then only reads the pixels of its block.
//
// For a Zarr image there is no shard per tile, so the correction needs "tile_grid": the size of
// the tile cell in the image and where that cell starts inside the field of view. Without it,
// Zarr data is served uncorrected.
//
// Channels marked as corrected at conversion (SAVAII's .conversion_metadata, key
// "flatfield_channels") must not get a profile here, or they would be corrected twice; the
// exporter in SAVAII leaves them out.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "json.hpp"

class FlatfieldModel
{
public:
    struct Profile
    {
        size_t h = 0, w = 0;
        std::vector<float> flat, dark;
    };

    struct Plane
    {
        size_t h = 0, w = 0;
        std::vector<float> inv_flat, dark; // 1 / max(flat, 1e-3), and dark (zero without darkfield)
    };

    // Persistence of materialised planes (set by the server, which has TensorStore): load returns
    // false when no stored plane matches; save is best effort.
    using PlaneIO = std::function<bool(const std::string &path, Plane &plane)>;
    static PlaneIO &load_hook()
    {
        static PlaneIO f;
        return f;
    }
    static PlaneIO &save_hook()
    {
        static PlaneIO f;
        return f;
    }

    std::string dir;     // <dataset>/.flatfield
    std::string version; // model version, part of the persisted plane names

    bool use_darkfield = true;
    bool has_grid = false;
    size_t cell_x = 0, cell_y = 0, window_x = 0, window_y = 0, fov_x = 0, fov_y = 0;

    // Returns nullptr when the dataset has no exported model.
    static FlatfieldModel *load(const std::string &dataset_dir)
    {
        const std::string dir = dataset_dir + "/.flatfield";
        std::ifstream index(dir + "/model.json");
        if (!index.good())
            return nullptr;
        try
        {
            nlohmann::json j = nlohmann::json::parse(index);
            auto *m = new FlatfieldModel();
            m->dir = dir;
            m->use_darkfield = j.value("use_darkfield", true);
            if (j.contains("version"))
                m->version = j["version"].is_string() ? j["version"].get<std::string>() : j["version"].dump();
            else
                m->version = std::to_string(std::hash<std::string>{}(j.dump()) & 0xffffffffffULL);
            if (j.contains("tile_grid"))
            {
                auto g = j["tile_grid"];
                m->cell_x = g["cell"][0];
                m->cell_y = g["cell"][1];
                m->window_x = g.value("window_origin", std::vector<size_t>{0, 0})[0];
                m->window_y = g.value("window_origin", std::vector<size_t>{0, 0})[1];
                auto fov = g.value("fov", std::vector<size_t>{m->cell_x + 2 * m->window_x,
                                                               m->cell_y + 2 * m->window_y});
                m->fov_x = fov[0];
                m->fov_y = fov[1];
                m->has_grid = m->cell_x > 0 && m->cell_y > 0;
            }
            for (auto &[key, ch] : j["channels"].items())
            {
                Profile p;
                p.h = ch["shape"][0];
                p.w = ch["shape"][1];
                if (!read_f32(dir + "/" + ch["flat"].get<std::string>(), p.h * p.w, p.flat))
                    continue;
                if (ch.contains("dark"))
                    read_f32(dir + "/" + ch["dark"].get<std::string>(), p.h * p.w, p.dark);
                if (p.dark.empty())
                    p.dark.assign(p.h * p.w, 0.0f);
                m->profiles[(size_t)std::stoul(key)] = std::move(p);
            }
            if (m->profiles.empty())
            {
                delete m;
                return nullptr;
            }
            return m;
        }
        catch (const std::exception &e)
        {
            return nullptr;
        }
    }

    bool has(size_t channel) const { return profiles.count(channel) > 0; }

    // Apply a materialised plane at pixel index i.
    static inline uint16_t apply(const Plane &pl, size_t i, uint16_t raw)
    {
        float v = (static_cast<float>(raw) - pl.dark[i]) * pl.inv_flat[i];
        v = std::min(std::max(v, 0.0f), 65535.0f);
        return static_cast<uint16_t>(v + 0.5f);
    }

    // Correct one raw value at (x, y) of a shard whose raw extent at this pyramid level is
    // (width, height). Convenience for single values; loops should hold plane() instead.
    uint16_t correct(size_t channel, size_t width, size_t height, size_t x, size_t y, uint16_t raw)
    {
        auto pl = plane(channel, width, height);
        if (!pl || x >= width || y >= height)
            return raw;
        return apply(*pl, y * width + x, raw);
    }

    // Zarr: the plane of a level (``scale``) and the mapping from image (gx, gy) to its pixels.
    struct GlobalView
    {
        std::shared_ptr<const Plane> plane;
        size_t cx = 1, cy = 1, wx = 0, wy = 0;
        inline size_t index(size_t gx, size_t gy) const
        {
            return ((gy % cy) + wy) * plane->w + (gx % cx) + wx;
        }
    };
    GlobalView global_view(size_t channel, size_t scale)
    {
        GlobalView v;
        if (!has_grid || !has(channel))
            return v;
        const size_t s = std::max<size_t>(1, scale);
        v.cx = std::max<size_t>(1, cell_x / s);
        v.cy = std::max<size_t>(1, cell_y / s);
        v.wx = window_x / s;
        v.wy = window_y / s;
        v.plane = plane(channel, std::max<size_t>(1, fov_x / s), std::max<size_t>(1, fov_y / s));
        if (v.plane && (v.cx + v.wx > v.plane->w || v.cy + v.wy > v.plane->h))
            v.plane = nullptr;
        return v;
    }

    // Zarr: correct a value at image position (gx, gy) of pyramid level `scale`, using the tile
    // grid from the model to find the position inside the field of view.
    uint16_t correct_global(size_t channel, size_t scale, size_t gx, size_t gy, uint16_t raw)
    {
        if (!has_grid || !has(channel))
            return raw;
        const size_t s = std::max<size_t>(1, scale);
        const size_t cx = std::max<size_t>(1, cell_x / s), cy = std::max<size_t>(1, cell_y / s);
        const size_t fov_w = std::max<size_t>(1, fov_x / s), fov_h = std::max<size_t>(1, fov_y / s);
        const size_t x = (gx % cx) + window_x / s;
        const size_t y = (gy % cy) + window_y / s;
        return correct(channel, fov_w, fov_h, x, y, raw);
    }

    // The materialised plane of a channel at a level size (thread safe, computed once).
    std::shared_ptr<const Plane> plane(size_t channel, size_t width, size_t height)
    {
        auto it_p = profiles.find(channel);
        if (it_p == profiles.end() || width == 0 || height == 0)
            return nullptr;
        const auto key = std::make_tuple(channel, width, height);
        std::lock_guard<std::mutex> lock(planes_mutex);
        auto it = planes.find(key);
        if (it != planes.end())
            return it->second;
        auto pl = std::make_shared<Plane>();
        pl->h = height;
        pl->w = width;
        const std::string path = dir + "/levels/c" + std::to_string(channel) + "_" + std::to_string(width) + "x" +
                                 std::to_string(height) + "_" + version + ".zarr";
        bool loaded = load_hook() && load_hook()(path, *pl) && pl->inv_flat.size() == width * height &&
                      pl->dark.size() == width * height;
        if (!loaded)
        {
            const Profile &p = it_p->second;
            std::vector<float> flat = resize(p.flat, p.h, p.w, height, width);
            pl->inv_flat.resize(flat.size());
            for (size_t i = 0; i < flat.size(); i++)
                pl->inv_flat[i] = 1.0f / std::max(flat[i], 1e-3f);
            pl->dark = use_darkfield ? resize(p.dark, p.h, p.w, height, width) : std::vector<float>(flat.size(), 0.0f);
            if (save_hook())
                save_hook()(path, *pl);
        }
        planes[key] = pl;
        return pl;
    }

    // Bilinear resize matching skimage.transform.resize(order=1, anti_aliasing=False), which is
    // what SAVAII uses to bring a profile to the frame size.
    static std::vector<float> resize(const std::vector<float> &src, size_t sh, size_t sw, size_t dh, size_t dw)
    {
        std::vector<float> out(dh * dw);
        const double ry = static_cast<double>(sh) / dh, rx = static_cast<double>(sw) / dw;
        for (size_t oy = 0; oy < dh; oy++)
        {
            double fy = std::min(std::max((oy + 0.5) * ry - 0.5, 0.0), static_cast<double>(sh - 1));
            size_t y0 = static_cast<size_t>(fy), y1 = std::min(y0 + 1, sh - 1);
            double wy = fy - y0;
            for (size_t ox = 0; ox < dw; ox++)
            {
                double fx = std::min(std::max((ox + 0.5) * rx - 0.5, 0.0), static_cast<double>(sw - 1));
                size_t x0 = static_cast<size_t>(fx), x1 = std::min(x0 + 1, sw - 1);
                double wx = fx - x0;
                out[oy * dw + ox] = static_cast<float>(
                    (1 - wy) * ((1 - wx) * src[y0 * sw + x0] + wx * src[y0 * sw + x1]) +
                    wy * ((1 - wx) * src[y1 * sw + x0] + wx * src[y1 * sw + x1]));
            }
        }
        return out;
    }

private:
    std::map<size_t, Profile> profiles;
    std::map<std::tuple<size_t, size_t, size_t>, std::shared_ptr<const Plane>> planes;
    std::mutex planes_mutex;

    static bool read_f32(const std::string &path, size_t count, std::vector<float> &out)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.good())
            return false;
        out.resize(count);
        f.read(reinterpret_cast<char *>(out.data()), count * sizeof(float));
        return static_cast<size_t>(f.gcount()) == count * sizeof(float);
    }

};
