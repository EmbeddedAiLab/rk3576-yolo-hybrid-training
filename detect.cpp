// detect.cpp - run the hybrid-trained detector (yolo_hybrid_*.bin) on one image
//
// Usage: ./detect weights.bin image.jpg [conf=0.30] [out=result.jpg]
// Env  : HYB_CPU=1 -> run in float on the CPU instead of the NPU (useful while training is using the NPU)
//
// Preprocessing, quantization, im2col and decode are identical to train_main.cpp.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <rknn_matmul_api.h>

constexpr int TILE = 256;
constexpr float LEAK = 0.1f, NMS_IOU = 0.45f;
static bool g_cpu_only = false;

static inline float sig(float x) { return 1.f / (1.f + std::exp(-x)); }
static inline int rup(int x, int a) { return (x + a - 1) / a * a; }
static inline int8_t q8(float x) {
    int q = int(std::lrintf(x));
    return int8_t(std::max(-127, std::min(127, q)));
}
bool ck(int r, const char* s) {
    if (r != RKNN_SUCC) std::fprintf(stderr, "%s failed: %d\n", s, r);
    return r == RKNN_SUCC;
}

struct MM {
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr io{};
    rknn_tensor_mem *a = nullptr, *b = nullptr, *c = nullptr;
    std::vector<int8_t> bhost;
    MM() = default;
    MM(const MM&) = delete;
    MM& operator=(const MM&) = delete;
    ~MM() {
        if (c) rknn_destroy_mem(ctx, c);
        if (b) rknn_destroy_mem(ctx, b);
        if (a) rknn_destroy_mem(ctx, a);
        if (ctx) rknn_matmul_destroy(ctx);
    }
    bool create(int M, int K, int N) {
        rknn_matmul_info i{};
        i.M = M; i.K = K; i.N = N;
        i.type = RKNN_INT8_MM_INT8_TO_INT32;
        i.B_layout = 0; i.B_quant_type = 0;
        i.AC_layout = 0; i.AC_quant_type = 0;
        if (!ck(rknn_matmul_create(&ctx, &i, &io), "matmul create")) return false;
        if (io.A.size != size_t(M) * K || io.B.size != size_t(K) * N || io.C.size != size_t(M) * N * 4) {
            std::fprintf(stderr, "Unexpected matmul buffer sizes\n");
            return false;
        }
        a = rknn_create_mem(ctx, io.A.size);
        b = rknn_create_mem(ctx, io.B.size);
        c = rknn_create_mem(ctx, io.C.size);
        return a && b && c && a->virt_addr && b->virt_addr && c->virt_addr;
    }
    bool run(const int8_t* A, int32_t* out) {
        std::memcpy(a->virt_addr, A, io.A.size);
        std::memcpy(b->virt_addr, bhost.data(), io.B.size);
        std::memset(c->virt_addr, 0, io.C.size);
        if (!ck(rknn_matmul_set_io_mem(ctx, a, &io.A), "bind A") ||
            !ck(rknn_matmul_set_io_mem(ctx, b, &io.B), "bind B") ||
            !ck(rknn_matmul_set_io_mem(ctx, c, &io.C), "bind C") ||
            !ck(rknn_mem_sync(ctx, a, RKNN_MEMORY_SYNC_TO_DEVICE), "sync A") ||
            !ck(rknn_mem_sync(ctx, b, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B") ||
            !ck(rknn_mem_sync(ctx, c, RKNN_MEMORY_SYNC_TO_DEVICE), "sync C") ||
            !ck(rknn_matmul_run(ctx), "matmul run") ||
            !ck(rknn_mem_sync(ctx, c, RKNN_MEMORY_SYNC_FROM_DEVICE), "sync output"))
            return false;
        std::memcpy(out, c->virt_addr, io.C.size);
        return true;
    }
};

struct Layer {
    int cin = 0, cout = 0, k = 0, s = 0, p = 0, act = 0;
    int hin = 0, win = 0, hout = 0, wout = 0, M = 0, K = 0, N = 0, Kp = 0, Np = 0;
    std::vector<float> w, b;
    float sw = 1.f;
    MM mm;
};

struct Model {
    int img = 0, nc = 0;
    std::vector<Layer> L;
    std::vector<std::string> names;
};

bool load_model(const char* path, Model& m) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }
    int32_t hdr[5];
    if (std::fread(hdr, 4, 5, f) != 5 || hdr[0] != 0x48594B52 || hdr[1] != 1) {
        std::cerr << "Bad weights file header\n"; std::fclose(f); return false;
    }
    m.img = hdr[2]; m.nc = hdr[3];
    m.L = std::vector<Layer>(hdr[4]);
    int h = m.img, w = m.img;
    for (Layer& l : m.L) {
        int32_t sp[6];
        if (std::fread(sp, 4, 6, f) != 6) { std::fclose(f); return false; }
        l.cin = sp[0]; l.cout = sp[1]; l.k = sp[2]; l.s = sp[3]; l.p = sp[4]; l.act = sp[5];
        l.hin = h; l.win = w;
        l.hout = (h + 2 * l.p - l.k) / l.s + 1;
        l.wout = (w + 2 * l.p - l.k) / l.s + 1;
        l.M = l.hout * l.wout; l.K = l.cin * l.k * l.k; l.N = l.cout;
        l.Kp = rup(l.K, 32); l.Np = rup(l.N, 32);
        l.w.resize(size_t(l.K) * l.N); l.b.resize(l.N);
        if (std::fread(l.w.data(), 4, l.w.size(), f) != l.w.size() ||
            std::fread(l.b.data(), 4, l.b.size(), f) != l.b.size()) {
            std::cerr << "Truncated weights file\n"; std::fclose(f); return false;
        }
        h = l.hout; w = l.wout;
    }
    for (int i = 0; i < m.nc; i++) {
        uint32_t len = 0;
        if (std::fread(&len, 4, 1, f) != 1 || len > 256) { std::fclose(f); return false; }
        std::string s(len, '\0');
        if (len && std::fread(&s[0], 1, len, f) != len) { std::fclose(f); return false; }
        m.names.push_back(s);
    }
    std::fclose(f);
    if (m.L.back().cout != 5 + m.nc) { std::cerr << "Head size mismatch\n"; return false; }

    if (!g_cpu_only) {
        for (Layer& l : m.L) {
            if (!l.mm.create(TILE, l.Kp, l.Np)) return false;
            float mx = 0;
            for (float x : l.w) mx = std::max(mx, std::fabs(x));
            l.sw = mx > 0 ? mx / 127.f : 1.f;
            l.mm.bhost.assign(size_t(l.Kp) * l.Np, 0);
            for (int k = 0; k < l.K; k++)
                for (int n = 0; n < l.N; n++)
                    l.mm.bhost[size_t(k) * l.Np + n] = q8(l.w[size_t(k) * l.N + n] / l.sw);
        }
    }
    return true;
}

bool layer_forward(Layer& L, const std::vector<float>& in, std::vector<float>& out) {
    const int K = L.K, N = L.N, M = L.M;
    std::vector<float> col(size_t(M) * K, 0.f);
    for (int oy = 0; oy < L.hout; oy++)
        for (int ox = 0; ox < L.wout; ox++) {
            float* row = &col[size_t(oy * L.wout + ox) * K];
            for (int ky = 0; ky < L.k; ky++) {
                int iy = oy * L.s - L.p + ky;
                if (iy < 0 || iy >= L.hin) continue;
                for (int kx = 0; kx < L.k; kx++) {
                    int ix = ox * L.s - L.p + kx;
                    if (ix < 0 || ix >= L.win) continue;
                    std::memcpy(row + (ky * L.k + kx) * L.cin, &in[size_t(iy * L.win + ix) * L.cin],
                                sizeof(float) * L.cin);
                }
            }
        }
    out.assign(size_t(M) * N, 0.f);
    auto activate = [&](float v) { return (L.act && v < 0.f) ? v * LEAK : v; };

    if (g_cpu_only) {
        for (int m = 0; m < M; m++) {
            float* o = &out[size_t(m) * N];
            for (int n = 0; n < N; n++) o[n] = L.b[n];
            for (int k = 0; k < K; k++) {
                float c = col[size_t(m) * K + k];
                if (c == 0.f) continue;
                for (int n = 0; n < N; n++) o[n] += c * L.w[size_t(k) * N + n];
            }
            for (int n = 0; n < N; n++) o[n] = activate(o[n]);
        }
        return true;
    }

    float mx = 0;
    for (float v : col) mx = std::max(mx, std::fabs(v));
    const float sx = mx > 0 ? mx / 127.f : 1.f, inv = 1.f / sx;
    std::vector<int8_t> aq(size_t(TILE) * L.Kp);
    std::vector<int32_t> acc(size_t(TILE) * L.Np);
    for (int t0 = 0; t0 < M; t0 += TILE) {
        int rows = std::min(TILE, M - t0);
        std::fill(aq.begin(), aq.end(), 0);
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < K; j++) aq[size_t(r) * L.Kp + j] = q8(col[size_t(t0 + r) * K + j] * inv);
        if (!L.mm.run(aq.data(), acc.data())) return false;
        for (int r = 0; r < rows; r++)
            for (int n = 0; n < N; n++)
                out[size_t(t0 + r) * N + n] = activate(acc[size_t(r) * L.Np + n] * sx * L.sw + L.b[n]);
    }
    return true;
}

struct Det { int cls; float score, x1, y1, x2, y2; };

static float iou(const Det& a, const Det& b) {
    float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1), ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// Same decode as train_main.cpp (coordinates normalized 0..1)
std::vector<Det> decode(const float* out, int S, int NC, float conf, float nms_iou) {
    const int N = 5 + NC;
    std::vector<Det> dets;
    for (int gy = 0; gy < S; gy++)
        for (int gx = 0; gx < S; gx++) {
            const float* o = out + size_t(gy * S + gx) * N;
            float so = sig(o[0]);
            if (so < conf * 0.5f) continue;
            float mx = *std::max_element(o + 5, o + 5 + NC), sum = 0, best = 0;
            int bc = 0;
            for (int k = 0; k < NC; k++) {
                float e = std::exp(o[5 + k] - mx);
                sum += e;
                if (e > best) { best = e; bc = k; }
            }
            float score = so * best / sum;
            if (score < conf) continue;
            float cx = (gx + sig(o[1])) / S, cy = (gy + sig(o[2])) / S, w = sig(o[3]), h = sig(o[4]);
            dets.push_back({bc, score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2});
        }
    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
    std::vector<Det> keep;
    for (const Det& d : dets) {
        bool ok = true;
        for (const Det& k : keep)
            if (k.cls == d.cls && iou(d, k) > nms_iou) { ok = false; break; }
        if (ok) keep.push_back(d);
    }
    return keep;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " weights.bin image.jpg [conf=0.30] [out=result.jpg]\n";
        return 1;
    }
    float conf = argc > 3 ? float(std::atof(argv[3])) : 0.30f;
    std::string outpath = argc > 4 ? argv[4] : "result.jpg";
    g_cpu_only = std::getenv("HYB_CPU") != nullptr;

    Model m;
    if (!load_model(argv[1], m)) return 1;
    std::cout << "Model: input " << m.img << "x" << m.img << ", " << m.nc << " classes, "
              << (g_cpu_only ? "CPU float" : "NPU INT8") << "\n";

    cv::Mat orig = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (orig.empty()) { std::cerr << "Cannot read image " << argv[2] << "\n"; return 1; }
    cv::Mat rs;
    cv::resize(orig, rs, cv::Size(m.img, m.img), 0, 0, cv::INTER_LINEAR);   // plain stretch, like training
    cv::cvtColor(rs, rs, cv::COLOR_BGR2RGB);
    std::vector<float> x(size_t(m.img) * m.img * 3), y;
    for (size_t i = 0; i < x.size(); i++) x[i] = rs.data[i] / 255.f;

    double t0 = (double)cv::getTickCount();
    for (Layer& l : m.L) {
        if (!layer_forward(l, x, y)) return 1;
        x.swap(y);
    }
    double ms = ((double)cv::getTickCount() - t0) / cv::getTickFrequency() * 1000.0;
    const int S = m.L.back().hout;

    // Top raw candidates regardless of threshold, to see what the model "thinks"
    std::vector<Det> top = decode(x.data(), S, m.nc, 0.0f, NMS_IOU);
    std::cout << "Inference: " << ms << " ms. Top raw candidates:\n";
    for (size_t i = 0; i < std::min<size_t>(3, top.size()); i++)
        std::printf("  %-10s score %.3f\n", m.names[top[i].cls].c_str(), top[i].score);

    std::vector<Det> dets = decode(x.data(), S, m.nc, conf, NMS_IOU);
    std::cout << dets.size() << " detection(s) at conf >= " << conf << "\n";
    for (const Det& d : dets) {
        int x1 = std::max(0, int(d.x1 * orig.cols)), y1 = std::max(0, int(d.y1 * orig.rows));
        int x2 = std::min(orig.cols - 1, int(d.x2 * orig.cols)), y2 = std::min(orig.rows - 1, int(d.y2 * orig.rows));
        std::printf("  %-10s %.2f  [%d,%d,%d,%d]\n", m.names[d.cls].c_str(), d.score, x1, y1, x2, y2);
        cv::Scalar col(0, 255, 0);
        cv::rectangle(orig, cv::Point(x1, y1), cv::Point(x2, y2), col, 2);
        char label[96];
        std::snprintf(label, sizeof(label), "%s %.2f", m.names[d.cls].c_str(), d.score);
        cv::putText(orig, label, cv::Point(x1, std::max(14, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 2);
    }
    if (!cv::imwrite(outpath, orig)) { std::cerr << "Failed to write " << outpath << "\n"; return 1; }
    std::cout << "Saved " << outpath << "\n";
    return 0;
}
