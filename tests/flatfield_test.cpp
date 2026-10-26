// Build and run: clang++ -std=c++17 -I src tests/flatfield_test.cpp -o /tmp/ff && /tmp/ff
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "flatfield.hpp"

static void write_f32(const std::string &path, const std::vector<float> &v)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
}

int main()
{
    namespace fs = std::filesystem;
    const std::string root = fs::temp_directory_path().string() + "/sisf_ff_test";
    fs::remove_all(root);
    fs::create_directories(root + "/.flatfield");

    // 4 x 8 profile (h, w): flat falls off toward the right edge, dark is 100
    const size_t h = 4, w = 8;
    std::vector<float> flat(h * w), dark(h * w, 100.0f);
    for (size_t y = 0; y < h; y++)
        for (size_t x = 0; x < w; x++)
            flat[y * w + x] = 1.0f - 0.05f * x;
    write_f32(root + "/.flatfield/c0_flat.f32", flat);
    write_f32(root + "/.flatfield/c0_dark.f32", dark);
    std::ofstream(root + "/.flatfield/model.json")
        << R"({"use_darkfield": true, "channels": {"0": {"flat": "c0_flat.f32", "dark": "c0_dark.f32", "shape": [4, 8]}}})";

    FlatfieldModel *m = FlatfieldModel::load(root);
    assert(m != nullptr);
    assert(m->has(0) && !m->has(1));

    // Same size as the profile: exact (raw - dark) / flat
    for (size_t x = 0; x < w; x++)
    {
        const uint16_t raw = 1100;
        const float expect = (1100.0f - 100.0f) / (1.0f - 0.05f * x);
        const uint16_t got = m->correct(0, w, h, x, 2, raw);
        assert(std::abs(got - std::round(expect)) <= 1);
    }

    // Pyramid level 2 (half size) samples the resized plane; values stay in range and monotone
    uint16_t prev = 0;
    for (size_t x = 0; x < w / 2; x++)
    {
        const uint16_t got = m->correct(0, w / 2, h / 2, x, 1, 1100);
        assert(got >= prev);
        prev = got;
    }

    // Clamping and pass-through
    assert(m->correct(0, w, h, 0, 0, 50) == 0);      // below dark
    assert(m->correct(1, w, h, 0, 0, 1234) == 1234); // channel without profile
    assert(m->correct(0, w, h, w, 0, 777) == 777);   // out of range

    // No model file: nullptr
    assert(FlatfieldModel::load(root + "/missing") == nullptr);

    // Resize matches skimage for an exact 2x upsample of a ramp at the centre pixels
    std::vector<float> ramp = {0.0f, 1.0f};
    auto up = FlatfieldModel::resize(ramp, 1, 2, 1, 4);
    assert(std::abs(up[0] - 0.0f) < 1e-6 && std::abs(up[1] - 0.25f) < 1e-6);
    assert(std::abs(up[2] - 0.75f) < 1e-6 && std::abs(up[3] - 1.0f) < 1e-6);

    delete m;
    fs::remove_all(root);
    std::cout << "flatfield_test: all checks passed" << std::endl;
    return 0;
}
