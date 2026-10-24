// Build and run: clang++ -std=c++17 -I src tests/zarr_ngff_test.cpp -o /tmp/zt && /tmp/zt
// Tests the Zarr v3 / OME-NGFF metadata parsing, the scale key mapping and the
// output byte order, without TensorStore.
#include <cassert>
#include <cstdio>
#include <iostream>

#include "zarr_ngff.hpp"

using namespace zarr_ngff;

static njson level_array(std::vector<int64_t> shape, std::vector<int64_t> shard, std::vector<int64_t> inner,
                         const char *dtype = "uint16",
                         std::vector<std::string> names = {"c", "z", "y", "x"})
{
    return njson{
        {"zarr_format", 3},
        {"node_type", "array"},
        {"shape", shape},
        {"data_type", dtype},
        {"dimension_names", names},
        {"chunk_grid", {{"name", "regular"}, {"configuration", {{"chunk_shape", shard}}}}},
        {"chunk_key_encoding", {{"name", "default"}}},
        {"fill_value", 0},
        {"codecs", njson::array({{{"name", "sharding_indexed"},
                                  {"configuration",
                                   {{"chunk_shape", inner},
                                    {"codecs", njson::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
                                                             {{"name", "zstd"}, {"configuration", {{"level", 3}}}}})},
                                    {"index_codecs", njson::array({{{"name", "bytes"}}, {{"name", "crc32c"}}})}}}}})},
    };
}

// Per-axis pyramid: level 1 halves x and y only, level 2 halves all three.
static njson root_group(bool with_keys, njson keys = njson())
{
    njson axes = njson::array({{{"name", "c"}, {"type", "channel"}},
                               {{"name", "z"}, {"type", "space"}, {"unit", "nanometer"}},
                               {{"name", "y"}, {"type", "space"}, {"unit", "nanometer"}},
                               {{"name", "x"}, {"type", "space"}, {"unit", "nanometer"}}});
    njson datasets = njson::array();
    const double scales[3][4] = {{1, 1000, 325.5, 325.5}, {1, 1000, 651, 651}, {1, 2000, 1302, 1302}};
    for (int i = 0; i < 3; i++)
    {
        datasets.push_back({{"path", std::to_string(i)},
                            {"coordinateTransformations",
                             njson::array({{{"type", "scale"},
                                            {"scale", {scales[i][0], scales[i][1], scales[i][2], scales[i][3]}}}})}});
    }
    njson g = {
        {"zarr_format", 3},
        {"node_type", "group"},
        {"attributes",
         {{"ome", {{"version", "0.5"}, {"multiscales", njson::array({{{"name", "canvas"}, {"axes", axes}, {"datasets", datasets}}})}}}}},
    };
    if (with_keys)
        g["attributes"]["stitchreg"] = {{"level_keys", keys}, {"layout", "per-axis"}};
    return g;
}

static void test_multiscale()
{
    std::string err;
    multiscale ms;
    assert(parse_multiscale_group(root_group(true, njson::array({"1", "2", "4"})), ms, err));
    assert(ms.keys_from_metadata);
    assert(ms.levels.size() == 3);
    assert(ms.levels[0].key == 1 && ms.levels[1].key == 2 && ms.levels[2].key == 4);
    assert(ms.axis_names.size() == 4 && ms.axis_names[0] == "c");

    std::vector<array_meta> arrays(3);
    assert(parse_array_meta(level_array({2, 40, 1000, 1200}, {1, 40, 512, 512}, {1, 40, 32, 32}), arrays[0], err));
    assert(parse_array_meta(level_array({2, 40, 500, 600}, {1, 40, 256, 256}, {1, 40, 32, 32}), arrays[1], err));
    assert(parse_array_meta(level_array({2, 20, 250, 300}, {1, 20, 128, 128}, {1, 20, 32, 32}), arrays[2], err));
    assert(arrays[0].sharded);
    assert(arrays[0].codecs.size() == 3 && arrays[0].codecs[0] == "sharding_indexed" && arrays[0].codecs[2] == "zstd");

    std::vector<level_view> v;
    if (!build_multiscale_views(ms, arrays, v, err))
    {
        std::cerr << err << std::endl;
        assert(false);
    }
    assert(v.size() == 3);
    assert(v[0].axes.c == 0 && v[0].axes.z == 1 && v[0].axes.y == 2 && v[0].axes.x == 3 && !v[0].axes.positional);
    assert(v[0].size_x == 1200 && v[0].size_y == 1000 && v[0].size_z == 40 && v[0].channels == 2);
    assert(v[0].res_nm_x == 325.5 && v[0].res_nm_y == 325.5 && v[0].res_nm_z == 1000);
    assert(v[1].size_x == 600 && v[1].size_z == 40 && v[1].res_nm_x == 651 && v[1].res_nm_z == 1000);
    assert(v[2].res_nm_z == 2000);
    assert(v[0].chunk_x == 512 && v[0].chunk_z == 40 && v[0].inner_x == 32 && v[0].inner_y == 32 && v[0].inner_z == 40);
    assert(v[0].dtype == sample_type::uint16);

    assert(find_level(v, 1) == 0 && find_level(v, 2) == 1 && find_level(v, 4) == 2 && find_level(v, 3) == -1);

    // Integer keys work too
    assert(parse_multiscale_group(root_group(true, njson::array({1, 2, 4})), ms, err) && ms.levels[2].key == 4);

    // Fallback to 2**i: absent, wrong length, duplicate, zero, non-numeric
    const njson bad[] = {njson::array({"1", "2"}), njson::array({"1", "1", "4"}), njson::array({"0", "2", "4"}),
                         njson::array({"a", "2", "4"})};
    assert(parse_multiscale_group(root_group(false), ms, err) && !ms.keys_from_metadata);
    assert(ms.levels[0].key == 1 && ms.levels[1].key == 2 && ms.levels[2].key == 4);
    for (const auto &k : bad)
    {
        assert(parse_multiscale_group(root_group(true, k), ms, err));
        assert(!ms.keys_from_metadata && ms.levels[1].key == 2 && ms.levels[2].key == 4);
    }

    // Custom keys are kept
    assert(parse_multiscale_group(root_group(true, njson::array({"1", "3", "9"})), ms, err) && ms.levels[1].key == 3);
}

static void test_units_and_dtype()
{
    std::string err;
    njson g = root_group(false);
    for (auto &a : g["attributes"]["ome"]["multiscales"][0]["axes"])
        if (a["type"] == "space")
            a["unit"] = "micrometer";
    multiscale ms;
    assert(parse_multiscale_group(g, ms, err));

    std::vector<array_meta> arrays(3);
    for (int i = 0; i < 3; i++)
        assert(parse_array_meta(level_array({1, 8, 64, 64}, {1, 8, 64, 64}, {1, 8, 32, 32}, "uint8"), arrays[i], err));
    std::vector<level_view> v;
    assert(build_multiscale_views(ms, arrays, v, err));
    assert(v[0].res_nm_z == 1000.0 * 1000.0 && v[0].res_nm_x == 325.5 * 1000.0);
    assert(v[0].dtype == sample_type::uint8 && v[0].channels == 1);

    // Unsupported dtype fails with a message naming it
    for (int i = 0; i < 3; i++)
        assert(parse_array_meta(level_array({1, 8, 64, 64}, {1, 8, 64, 64}, {1, 8, 32, 32}, "float32"), arrays[i], err));
    assert(!build_multiscale_views(ms, arrays, v, err));
    assert(err.find("float32") != std::string::npos);

    // Unknown unit fails
    g["attributes"]["ome"]["multiscales"][0]["axes"][3]["unit"] = "parsec";
    assert(parse_multiscale_group(g, ms, err));
    for (int i = 0; i < 3; i++)
        assert(parse_array_meta(level_array({1, 8, 64, 64}, {1, 8, 64, 64}, {1, 8, 32, 32}), arrays[i], err));
    assert(!build_multiscale_views(ms, arrays, v, err));
}

static void test_axes()
{
    std::string err;
    axis_map m;

    // Legacy plain array without names: x, y, z, c by position
    assert(resolve_axes({}, 4, m, err) && m.positional && m.x == 0 && m.y == 1 && m.z == 2 && m.c == 3);
    assert(resolve_axes({"", "", ""}, 3, m, err) && m.positional && m.c == -1);
    assert(!resolve_axes({}, 5, m, err));

    // Named axes, any order, extra axes go to "other"
    assert(resolve_axes({"t", "c", "z", "y", "x"}, 5, m, err));
    assert(m.x == 4 && m.y == 3 && m.z == 2 && m.c == 1 && m.other.size() == 1 && m.other[0] == 0);
    assert(resolve_axes({"X", "Y", "Z"}, 3, m, err) && m.x == 0 && m.c == -1);
    assert(!resolve_axes({"x", "x", "z"}, 3, m, err));
    assert(!resolve_axes({"c", "y", "x"}, 3, m, err));

    // A group that is not NGFF is rejected, a plain array is not a group
    multiscale ms;
    assert(!parse_multiscale_group(njson{{"zarr_format", 3}, {"node_type", "group"}, {"attributes", njson::object()}}, ms, err));
    assert(!parse_multiscale_group(level_array({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}), ms, err));

    // Unsharded array: inner chunk equals the grid chunk
    array_meta a;
    njson plain = {{"zarr_format", 3}, {"node_type", "array"}, {"shape", {100, 80, 6, 2}}, {"data_type", "uint16"},
                   {"chunk_grid", {{"name", "regular"}, {"configuration", {{"chunk_shape", {64, 64, 6, 1}}}}}},
                   {"codecs", njson::array({{{"name", "bytes"}}})}};
    assert(parse_array_meta(plain, a, err) && !a.sharded && a.inner_chunk_shape == a.chunk_shape);
    level_view v;
    v.array = a;
    assert(resolve_axes(a.dimension_names, a.shape.size(), v.axes, err));
    assert(fill_level_view(v, err));
    assert(v.size_x == 100 && v.size_y == 80 && v.size_z == 6 && v.channels == 2 && v.chunk_x == 64);
}

// Value of logical sample (c, z, y, x)
static uint16_t val(size_t c, size_t z, size_t y, size_t x) { return (uint16_t)(c * 1000 + z * 100 + y * 10 + x); }

// SISF block layout, as in archive_reader::load_region's SISF branch
static size_t sisf_offset(size_t c, size_t z, size_t y, size_t x, size_t nz, size_t ny, size_t nx)
{
    return (c * ny * nx * nz) + (z * ny * nx) + (y * nx) + x;
}

static void test_byte_order()
{
    const size_t nc = 2, nz = 3, ny = 4, nx = 5;
    std::string err;

    // Source in (c, z, y, x) C order, as an NGFF array
    {
        std::vector<uint16_t> src(nc * nz * ny * nx);
        for (size_t c = 0; c < nc; c++)
            for (size_t z = 0; z < nz; z++)
                for (size_t y = 0; y < ny; y++)
                    for (size_t x = 0; x < nx; x++)
                        src[((c * nz + z) * ny + y) * nx + x] = val(c, z, y, x);
        axis_map m;
        assert(resolve_axes({"c", "z", "y", "x"}, 4, m, err));
        std::vector<int64_t> bs = {(int64_t)(nz * ny * nx * 2), (int64_t)(ny * nx * 2), (int64_t)(nx * 2), 2};
        std::vector<uint16_t> out(src.size(), 0xFFFF);
        copy_region_to_czyx_u16(src.data(), bs, m, nc, nz, ny, nx, out.data());
        for (size_t c = 0; c < nc; c++)
            for (size_t z = 0; z < nz; z++)
                for (size_t y = 0; y < ny; y++)
                    for (size_t x = 0; x < nx; x++)
                        assert(out[sisf_offset(c, z, y, x, nz, ny, nx)] == val(c, z, y, x));
    }

    // Source in (x, y, z, c) C order, as the legacy positional array
    {
        std::vector<uint16_t> src(nc * nz * ny * nx);
        for (size_t c = 0; c < nc; c++)
            for (size_t z = 0; z < nz; z++)
                for (size_t y = 0; y < ny; y++)
                    for (size_t x = 0; x < nx; x++)
                        src[((x * ny + y) * nz + z) * nc + c] = val(c, z, y, x);
        axis_map m;
        assert(resolve_axes({}, 4, m, err));
        std::vector<int64_t> bs = {(int64_t)(ny * nz * nc * 2), (int64_t)(nz * nc * 2), (int64_t)(nc * 2), 2};
        std::vector<uint16_t> out(src.size(), 0xFFFF);
        copy_region_to_czyx_u16(src.data(), bs, m, nc, nz, ny, nx, out.data());
        for (size_t c = 0; c < nc; c++)
            for (size_t z = 0; z < nz; z++)
                for (size_t y = 0; y < ny; y++)
                    for (size_t x = 0; x < nx; x++)
                        assert(out[sisf_offset(c, z, y, x, nz, ny, nx)] == val(c, z, y, x));
    }

    // uint8 (t, c, z, y, x) with t extent 1, widened to uint16
    {
        std::vector<uint8_t> src(nc * nz * ny * nx);
        for (size_t c = 0; c < nc; c++)
            for (size_t z = 0; z < nz; z++)
                for (size_t y = 0; y < ny; y++)
                    for (size_t x = 0; x < nx; x++)
                        src[((c * nz + z) * ny + y) * nx + x] = (uint8_t)(val(c, z, y, x) % 251);
        axis_map m;
        assert(resolve_axes({"t", "c", "z", "y", "x"}, 5, m, err));
        std::vector<int64_t> bs = {(int64_t)(nc * nz * ny * nx), (int64_t)(nz * ny * nx), (int64_t)(ny * nx), (int64_t)nx, 1};
        std::vector<uint16_t> out(src.size(), 0xFFFF);
        copy_region_to_czyx_u16(src.data(), bs, m, nc, nz, ny, nx, out.data());
        for (size_t c = 0; c < nc; c++)
            for (size_t z = 0; z < nz; z++)
                for (size_t y = 0; y < ny; y++)
                    for (size_t x = 0; x < nx; x++)
                        assert(out[sisf_offset(c, z, y, x, nz, ny, nx)] == val(c, z, y, x) % 251);
    }

    // Single channel without a c axis
    {
        std::vector<uint16_t> src(nz * ny * nx);
        for (size_t z = 0; z < nz; z++)
            for (size_t y = 0; y < ny; y++)
                for (size_t x = 0; x < nx; x++)
                    src[(z * ny + y) * nx + x] = val(0, z, y, x);
        axis_map m;
        assert(resolve_axes({"z", "y", "x"}, 3, m, err) && m.c == -1);
        std::vector<int64_t> bs = {(int64_t)(ny * nx * 2), (int64_t)(nx * 2), 2};
        std::vector<uint16_t> out(src.size(), 0xFFFF);
        copy_region_to_czyx_u16(src.data(), bs, m, 1, nz, ny, nx, out.data());
        assert(out == src);
    }
}

int main()
{
    test_multiscale();
    test_units_and_dtype();
    test_axes();
    test_byte_order();
    std::cout << "zarr_ngff_test: all passed" << std::endl;
    return 0;
}
