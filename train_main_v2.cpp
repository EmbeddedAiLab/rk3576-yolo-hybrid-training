// train_main.cpp - hybrid NPU+CPU training of a small YOLO-style grid detector on RK3576
//
//  NPU : INT8 matmul for every conv layer's forward pass (im2col -> rknn_matmul)
//  CPU : im2col, dequant, loss, backprop (float, straight-through estimator), SGD+momentum
//
// Model (v2): 8 conv layers, stride 16, S x S grid (S = IMG/16), one box per cell
//   per cell output = [obj, tx, ty, tw, th, class logits...]
//   box = ((gx+sig(tx))/S, (gy+sig(ty))/S, sig(tw), sig(th))   (all relative to image)
//
// Usage: ./train_main [dataset_dir=dataset] [epochs=30] [lr=0.01] [max_train_images=0(all)]
// Output: yolo_hybrid_last.bin, yolo_hybrid_best.bin (best val F1)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <rknn_matmul_api.h>

namespace fs = std::filesystem;

constexpr int IMG = 192;          // network input (square, stretched resize). Must be a multiple of STRIDE.
constexpr int STRIDE = 16;        // total downsampling of the network
constexpr int S = IMG / STRIDE;   // grid size (12x12)
constexpr int TILE = 256;         // rows of A per NPU matmul call
constexpr int BATCH = 8;
constexpr float MOMENTUM = 0.9f, WD = 5e-4f, CLIP = 10.f;
constexpr float LBOX = 5.f, WNEG = 0.05f, LEAK = 0.1f;
constexpr float EVAL_CONF = 0.30f, NMS_IOU = 0.45f;

static bool g_cpu_only = false;   // HYB_CPU=1: bypass NPU, float matmul on CPU
static int g_check = 0;           // HYB_CHECK=1: compare NPU vs float for first layers

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

// ---------------------------------------------------------------- NPU matmul
struct MM {
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr io{};
    rknn_tensor_mem *a = nullptr, *b = nullptr, *c = nullptr;
    int m = 0, k = 0, n = 0;
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
        m = M; k = K; n = N;
        rknn_matmul_info i{};
        i.M = M; i.K = K; i.N = N;
        i.type = RKNN_INT8_MM_INT8_TO_INT32;
        i.B_layout = 0; i.B_quant_type = 0;
        i.AC_layout = 0; i.AC_quant_type = 0;
        if (!ck(rknn_matmul_create(&ctx, &i, &io), "matmul create")) return false;
        if (io.A.size != size_t(M) * K || io.B.size != size_t(K) * N ||
            io.C.size != size_t(M) * N * 4) {
            std::fprintf(stderr, "Unexpected matmul buffer sizes (M=%d K=%d N=%d): A=%u B=%u C=%u\n",
                         M, K, N, io.A.size, io.B.size, io.C.size);
            return false;
        }
        a = rknn_create_mem(ctx, io.A.size);
        b = rknn_create_mem(ctx, io.B.size);
        c = rknn_create_mem(ctx, io.C.size);
        if (!a || !b || !c || !a->virt_addr || !b->virt_addr || !c->virt_addr) {
            std::fprintf(stderr, "NPU buffer allocation failed\n");
            return false;
        }
        std::memset(a->virt_addr, 0, io.A.size);
        std::memset(b->virt_addr, 0, io.B.size);
        std::memset(c->virt_addr, 0, io.C.size);
        return true;
    }
    std::vector<int8_t> bhost;
    bool set_weights(const int8_t* w) {          // keep a host copy; uploaded on every run (same as train_hybrid.cpp)
        bhost.assign(w, w + io.B.size);
        return true;
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

// ---------------------------------------------------------------- conv layer
struct Spec { int cin, cout, k, s, p; bool act; };

struct Layer {
    Spec sp{};
    int hin = 0, win = 0, hout = 0, wout = 0, M = 0, K = 0, N = 0, Kp = 0, Np = 0;
    std::vector<float> w, b, gw, gb, vw, vb;      // w is [K][N]
    std::vector<int8_t> wq, aq;
    std::vector<int32_t> acc;
    float sw = 1.f, sx = 1.f;
    std::vector<float> col, pre, out, dz;          // per-image cache (HWC layout)
    MM mm;
};

bool layer_init(Layer& L, const Spec& sp, int hin, int win, bool head, std::mt19937& rng) {
    L.sp = sp; L.hin = hin; L.win = win;
    L.hout = (hin + 2 * sp.p - sp.k) / sp.s + 1;
    L.wout = (win + 2 * sp.p - sp.k) / sp.s + 1;
    L.M = L.hout * L.wout;
    L.K = sp.cin * sp.k * sp.k;
    L.N = sp.cout;
    L.Kp = rup(L.K, 32);
    L.Np = rup(L.N, 32);
    L.w.resize(size_t(L.K) * L.N);
    L.b.assign(L.N, 0.f);
    L.gw.assign(L.w.size(), 0.f); L.gb.assign(L.N, 0.f);
    L.vw.assign(L.w.size(), 0.f); L.vb.assign(L.N, 0.f);
    float stdv = head ? 0.01f : std::sqrt(2.f / L.K);
    std::normal_distribution<float> nd(0.f, stdv);
    for (float& x : L.w) x = nd(rng);
    if (head) L.b[0] = -4.f;                       // objectness prior: start with "nothing here"
    L.wq.assign(size_t(L.Kp) * L.Np, 0);
    return L.mm.create(TILE, L.Kp, L.Np);
}

bool quantize_weights(Layer& L) {
    float m = 0;
    for (float x : L.w) m = std::max(m, std::fabs(x));
    L.sw = m > 0 ? m / 127.f : 1.f;
    std::fill(L.wq.begin(), L.wq.end(), 0);
    for (int k = 0; k < L.K; k++)
        for (int n = 0; n < L.N; n++)
            L.wq[size_t(k) * L.Np + n] = q8(L.w[size_t(k) * L.N + n] / L.sw);
    return L.mm.set_weights(L.wq.data());
}

bool layer_forward(Layer& L, const float* in) {
    const int K = L.K, N = L.N, M = L.M, k = L.sp.k, cin = L.sp.cin;
    L.col.assign(size_t(M) * K, 0.f);
    for (int oy = 0; oy < L.hout; oy++)
        for (int ox = 0; ox < L.wout; ox++) {
            float* row = &L.col[size_t(oy * L.wout + ox) * K];
            for (int ky = 0; ky < k; ky++) {
                int iy = oy * L.sp.s - L.sp.p + ky;
                if (iy < 0 || iy >= L.hin) continue;
                for (int kx = 0; kx < k; kx++) {
                    int ix = ox * L.sp.s - L.sp.p + kx;
                    if (ix < 0 || ix >= L.win) continue;
                    std::memcpy(row + (ky * k + kx) * cin,
                                in + size_t(iy * L.win + ix) * cin, sizeof(float) * cin);
                }
            }
        }
    float mx = 0;
    for (float v : L.col) mx = std::max(mx, std::fabs(v));
    L.sx = mx > 0 ? mx / 127.f : 1.f;              // dynamic per-tensor activation scale
    const float inv = 1.f / L.sx;

    L.pre.assign(size_t(M) * N, 0.f);
    L.out.assign(size_t(M) * N, 0.f);
    if (g_cpu_only) {
#pragma omp parallel for
        for (int m = 0; m < M; m++) {
            float* o = &L.pre[size_t(m) * N];
            for (int n = 0; n < N; n++) o[n] = L.b[n];
            for (int kk = 0; kk < K; kk++) {
                float c = L.col[size_t(m) * K + kk];
                if (c == 0.f) continue;
                const float* w = &L.w[size_t(kk) * N];
                for (int n = 0; n < N; n++) o[n] += c * w[n];
            }
            for (int n = 0; n < N; n++) {
                float v = o[n];
                L.out[size_t(m) * N + n] = (L.sp.act && v < 0.f) ? v * LEAK : v;
            }
        }
        return true;
    }
    L.aq.resize(size_t(TILE) * L.Kp);
    L.acc.resize(size_t(TILE) * L.Np);
    for (int t0 = 0; t0 < M; t0 += TILE) {
        int rows = std::min(TILE, M - t0);
        std::fill(L.aq.begin(), L.aq.end(), 0);
        for (int r = 0; r < rows; r++) {
            const float* src = &L.col[size_t(t0 + r) * K];
            int8_t* dst = &L.aq[size_t(r) * L.Kp];
            for (int j = 0; j < K; j++) dst[j] = q8(src[j] * inv);
        }
        if (!L.mm.run(L.aq.data(), L.acc.data())) return false;
        for (int r = 0; r < rows; r++)
            for (int n = 0; n < N; n++) {
                float v = L.acc[size_t(r) * L.Np + n] * L.sx * L.sw + L.b[n];
                size_t o = size_t(t0 + r) * N + n;
                L.pre[o] = v;
                L.out[o] = (L.sp.act && v < 0.f) ? v * LEAK : v;
            }
    }
    if (g_check > 0) {
        double num = 0, den = 0, dot = 0, na = 0;
        std::vector<float> ref(N);
        for (int m = 0; m < M; m++) {
            std::fill(ref.begin(), ref.end(), 0.f);
            for (int kk = 0; kk < K; kk++) {
                float c = L.col[size_t(m) * K + kk];
                if (c == 0.f) continue;
                for (int n = 0; n < N; n++) ref[n] += c * L.w[size_t(kk) * N + n];
            }
            for (int n = 0; n < N; n++) {
                double a = L.pre[size_t(m) * N + n] - L.b[n], r = ref[n];
                num += (a - r) * (a - r); den += r * r; dot += a * r; na += a * a;
            }
        }
        std::printf("  [check] M=%d K=%d N=%d  relerr %.4f  corr %.4f  rms_float %.4f  rms_npu %.4f\n",
                    M, K, N, std::sqrt(num / std::max(den, 1e-12)), dot / std::sqrt(std::max(na * den, 1e-12)),
                    std::sqrt(den / (double(M) * N)), std::sqrt(na / (double(M) * N)));
        g_check--;
    }
    return true;
}

// Float backward (STE through the quantizer). Accumulates into gw/gb.
// din (nullable) must be zeroed, size hin*win*cin.
void layer_backward(Layer& L, const float* dout, float* din) {
    const int K = L.K, N = L.N, M = L.M, k = L.sp.k, cin = L.sp.cin;
    L.dz.resize(size_t(M) * N);
    for (size_t i = 0; i < L.dz.size(); i++)
        L.dz[i] = L.sp.act ? dout[i] * (L.pre[i] > 0.f ? 1.f : LEAK) : dout[i];
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) L.gb[n] += L.dz[size_t(m) * N + n];

#pragma omp parallel for
    for (int kk = 0; kk < K; kk++) {
        float* g = &L.gw[size_t(kk) * N];
        for (int m = 0; m < M; m++) {
            float c = L.col[size_t(m) * K + kk];
            if (c == 0.f) continue;
            const float* d = &L.dz[size_t(m) * N];
            for (int n = 0; n < N; n++) g[n] += c * d[n];
        }
    }
    if (!din) return;

    std::vector<float> dcol(size_t(M) * K);
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const float* d = &L.dz[size_t(m) * N];
        for (int kk = 0; kk < K; kk++) {
            const float* w = &L.w[size_t(kk) * N];
            float s = 0.f;
            for (int n = 0; n < N; n++) s += d[n] * w[n];
            dcol[size_t(m) * K + kk] = s;
        }
    }
    for (int oy = 0; oy < L.hout; oy++)
        for (int ox = 0; ox < L.wout; ox++) {
            const float* row = &dcol[size_t(oy * L.wout + ox) * K];
            for (int ky = 0; ky < k; ky++) {
                int iy = oy * L.sp.s - L.sp.p + ky;
                if (iy < 0 || iy >= L.hin) continue;
                for (int kx = 0; kx < k; kx++) {
                    int ix = ox * L.sp.s - L.sp.p + kx;
                    if (ix < 0 || ix >= L.win) continue;
                    float* dst = din + size_t(iy * L.win + ix) * cin;
                    const float* src = row + (ky * k + kx) * cin;
                    for (int c = 0; c < cin; c++) dst[c] += src[c];
                }
            }
        }
}

typedef std::vector<Layer> Net;

bool net_forward(Net& net, const float* img) {
    const float* x = img;
    for (Layer& l : net) {
        if (!layer_forward(l, x)) return false;
        x = l.out.data();
    }
    return true;
}

void net_backward(Net& net, const std::vector<float>& dhead) {
    std::vector<float> d = dhead, din;
    for (int i = int(net.size()) - 1; i >= 0; i--) {
        Layer& l = net[i];
        if (i > 0) din.assign(size_t(l.hin) * l.win * l.sp.cin, 0.f);
        layer_backward(l, d.data(), i > 0 ? din.data() : nullptr);
        if (i > 0) d.swap(din);
    }
}

bool quantize_all(Net& net) {
    if (g_cpu_only) return true;
    for (Layer& l : net) if (!quantize_weights(l)) return false;
    return true;
}

// ---------------------------------------------------------------- dataset
struct Box { int cls; float cx, cy, w, h; };
struct Sample { std::vector<uint8_t> rgb; std::vector<Box> boxes; };

static std::string trim(std::string s) {
    const char* ws = " \t\r\n\"'";
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

std::vector<std::string> parse_names(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::string> names;
    std::string line;
    bool in = false;
    while (std::getline(f, line)) {
        if (!in) {
            size_t p = line.find("names:");
            if (p == std::string::npos || line.find('#') < p) continue;
            std::string rest = line.substr(p + 6);
            size_t lb = rest.find('[');
            if (lb != std::string::npos) {                       // names: ['a','b']
                size_t rb = rest.find(']');
                std::string inner = rest.substr(lb + 1, rb - lb - 1), tok;
                std::stringstream ss(inner);
                while (std::getline(ss, tok, ',')) names.push_back(trim(tok));
                return names;
            }
            in = true;
            continue;
        }
        if (line.empty() || line[0] == '#') continue;
        if (line[0] != ' ' && line[0] != '\t' && line[0] != '-') break;   // next yaml key
        std::string s = trim(line);
        if (s.empty()) continue;
        if (s[0] == '-') names.push_back(trim(s.substr(1)));              // - name
        else { size_t c = s.find(':'); if (c != std::string::npos) names.push_back(trim(s.substr(c + 1))); } // 0: name
    }
    return names;
}

bool load_split(const std::string& root, const std::string& split, int NC, int limit,
                std::vector<Sample>& out) {
    fs::path imd = fs::path(root) / "images" / split, lbd = fs::path(root) / "labels" / split;
    if (!fs::exists(imd)) { std::cerr << "Missing " << imd << "\n"; return false; }
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(imd)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (limit > 0 && int(files.size()) > limit) files.resize(limit);
    for (auto& p : files) {
        cv::Mat im = cv::imread(p.string(), cv::IMREAD_COLOR);
        if (im.empty()) continue;
        cv::resize(im, im, cv::Size(IMG, IMG), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
        Sample s;
        s.rgb.assign(im.data, im.data + size_t(IMG) * IMG * 3);
        std::ifstream lf(lbd / (p.stem().string() + ".txt"));
        Box b;
        while (lf >> b.cls >> b.cx >> b.cy >> b.w >> b.h)
            if (b.cls >= 0 && b.cls < NC && b.w > 0 && b.h > 0) s.boxes.push_back(b);
        out.push_back(std::move(s));
    }
    return !out.empty();
}

void make_input(const Sample& s, bool flip, float gain, std::vector<float>& x) {
    x.resize(size_t(IMG) * IMG * 3);
    for (int y = 0; y < IMG; y++)
        for (int xx = 0; xx < IMG; xx++) {
            int sx = flip ? IMG - 1 - xx : xx;
            for (int c = 0; c < 3; c++)
                x[(size_t(y) * IMG + xx) * 3 + c] =
                    std::min(1.f, s.rgb[(size_t(y) * IMG + sx) * 3 + c] / 255.f * gain);
        }
}

// Random zoom/shift/flip + color jitter on the stored IMGxIMG image; boxes are transformed to match.
void augment(const Sample& s, std::mt19937& rng, std::vector<float>& x, std::vector<Box>& boxes) {
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float sc = 0.6f + 0.9f * U(rng);
    const float tx = (U(rng) - 0.5f) * 0.4f, ty = (U(rng) - 0.5f) * 0.4f;
    const bool flip = U(rng) < 0.5f;
    cv::Mat src(IMG, IMG, CV_8UC3, const_cast<uint8_t*>(s.rgb.data())), dst;
    cv::Mat M = (cv::Mat_<double>(2, 3) << sc, 0, IMG * (0.5 - 0.5 * sc + tx), 0, sc, IMG * (0.5 - 0.5 * sc + ty));
    cv::warpAffine(src, dst, M, cv::Size(IMG, IMG), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (flip) cv::flip(dst, dst, 1);
    const float gain = 0.7f + 0.6f * U(rng);
    const float cg[3] = {0.9f + 0.2f * U(rng), 0.9f + 0.2f * U(rng), 0.9f + 0.2f * U(rng)};
    x.resize(size_t(IMG) * IMG * 3);
    for (size_t i = 0; i < x.size(); i++) x[i] = std::min(1.f, dst.data[i] / 255.f * gain * cg[i % 3]);
    boxes.clear();
    for (const Box& b : s.boxes) {
        float cx = (b.cx - 0.5f) * sc + 0.5f + tx, cy = (b.cy - 0.5f) * sc + 0.5f + ty;
        float w = b.w * sc, h = b.h * sc;
        float x1 = std::max(0.f, cx - w / 2), x2 = std::min(1.f, cx + w / 2);
        float y1 = std::max(0.f, cy - h / 2), y2 = std::min(1.f, cy + h / 2);
        float nw = x2 - x1, nh = y2 - y1;
        if (nw < 0.02f || nh < 0.02f || nw * nh < 0.4f * w * h) continue;   // mostly cropped out
        Box nb{b.cls, (x1 + x2) / 2, (y1 + y2) / 2, nw, nh};
        if (flip) nb.cx = 1.f - nb.cx;
        boxes.push_back(nb);
    }
}

// ---------------------------------------------------------------- loss / decode
struct Tgt { bool pos = false; float tx = 0, ty = 0, w = 0, h = 0; int cls = 0; };

void build_targets(const std::vector<Box>& boxes, bool flip, std::vector<Tgt>& t) {
    t.assign(S * S, Tgt());
    for (const Box& b : boxes) {
        float cx = flip ? 1.f - b.cx : b.cx;
        int gx = std::min(S - 1, std::max(0, int(cx * S)));
        int gy = std::min(S - 1, std::max(0, int(b.cy * S)));
        Tgt& c = t[gy * S + gx];
        if (c.pos && c.w * c.h >= b.w * b.h) continue;   // keep the larger object in a shared cell
        c.pos = true; c.tx = cx * S - gx; c.ty = b.cy * S - gy;
        c.w = std::min(1.f, b.w); c.h = std::min(1.f, b.h); c.cls = b.cls;
    }
}

double head_loss(const float* out, const std::vector<Tgt>& t, int NC, std::vector<float>* d) {
    const int N = 5 + NC;
    if (d) d->assign(size_t(S) * S * N, 0.f);
    double loss = 0;
    std::vector<float> p(NC);
    for (int c = 0; c < S * S; c++) {
        const float* o = out + size_t(c) * N;
        float so = sig(o[0]);
        const Tgt& g = t[c];
        if (!g.pos) {
            loss += -WNEG * std::log(1.f - so + 1e-7f);
            if (d) (*d)[size_t(c) * N] = WNEG * so;
            continue;
        }
        loss += -std::log(so + 1e-7f);
        if (d) (*d)[size_t(c) * N] = so - 1.f;
        const float tg[4] = {g.tx, g.ty, g.w, g.h};
        for (int j = 0; j < 4; j++) {
            float pj = sig(o[1 + j]), e = pj - tg[j];
            loss += LBOX * e * e;
            if (d) (*d)[size_t(c) * N + 1 + j] = LBOX * 2.f * e * pj * (1.f - pj);
        }
        float mx = *std::max_element(o + 5, o + 5 + NC), sum = 0;
        for (int k = 0; k < NC; k++) { p[k] = std::exp(o[5 + k] - mx); sum += p[k]; }
        for (int k = 0; k < NC; k++) {
            p[k] /= sum;
            if (d) (*d)[size_t(c) * N + 5 + k] = p[k] - (k == g.cls ? 1.f : 0.f);
        }
        loss += -std::log(std::max(p[g.cls], 1e-7f));
    }
    return loss;
}

struct Det { int cls; float score, x1, y1, x2, y2; };

static float iou(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
    float iw = std::min(ax2, bx2) - std::max(ax1, bx1), ih = std::min(ay2, by2) - std::max(ay1, by1);
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter);
}

// Same decode must be used in detect.cpp
std::vector<Det> decode(const float* out, int NC, float conf, float nms_iou) {
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
            if (k.cls == d.cls && iou(d.x1, d.y1, d.x2, d.y2, k.x1, k.y1, k.x2, k.y2) > nms_iou) { ok = false; break; }
        if (ok) keep.push_back(d);
    }
    return keep;
}

struct EvalRes { double loss = 0, P = 0, R = 0, F1 = 0; };

bool evaluate(Net& net, const std::vector<Sample>& val, int NC, EvalRes& r) {
    if (!quantize_all(net)) return false;
    int tp = 0, fp = 0, gt = 0;
    double loss = 0;
    std::vector<float> x;
    std::vector<Tgt> t;
    for (const Sample& s : val) {
        make_input(s, false, 1.f, x);
        if (!net_forward(net, x.data())) return false;
        build_targets(s.boxes, false, t);
        loss += head_loss(net.back().out.data(), t, NC, nullptr);
        std::vector<Det> dets = decode(net.back().out.data(), NC, EVAL_CONF, NMS_IOU);
        std::vector<bool> used(s.boxes.size(), false);
        gt += int(s.boxes.size());
        for (const Det& d : dets) {
            int best = -1; float bi = 0.5f;
            for (size_t j = 0; j < s.boxes.size(); j++) {
                const Box& b = s.boxes[j];
                if (used[j] || b.cls != d.cls) continue;
                float v = iou(d.x1, d.y1, d.x2, d.y2, b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2);
                if (v >= bi) { bi = v; best = int(j); }
            }
            if (best >= 0) { used[best] = true; tp++; } else fp++;
        }
    }
    r.loss = loss / std::max<size_t>(1, val.size());
    r.P = tp + fp ? double(tp) / (tp + fp) : 0;
    r.R = gt ? double(tp) / gt : 0;
    r.F1 = (r.P + r.R) > 0 ? 2 * r.P * r.R / (r.P + r.R) : 0;
    return true;
}

// ---------------------------------------------------------------- save
// File layout (little endian):
//   int32 magic 'RKYH', version(1), IMG, NC, nlayers
//   per layer: int32 cin,cout,k,s,p,act ; float w[K*N] (K=cin*k*k, index k*N+n, k=(ky*kx_idx)*cin+c) ; float b[N]
//   per class: uint32 len, chars
bool save_bin(const std::string& path, Net& net, const std::vector<std::string>& names) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    int32_t hdr[5] = {0x48594B52, 1, IMG, int32_t(names.size()), int32_t(net.size())};
    std::fwrite(hdr, 4, 5, f);
    for (Layer& l : net) {
        int32_t sp[6] = {l.sp.cin, l.sp.cout, l.sp.k, l.sp.s, l.sp.p, l.sp.act ? 1 : 0};
        std::fwrite(sp, 4, 6, f);
        std::fwrite(l.w.data(), 4, l.w.size(), f);
        std::fwrite(l.b.data(), 4, l.b.size(), f);
    }
    for (const std::string& n : names) {
        uint32_t len = uint32_t(n.size());
        std::fwrite(&len, 4, 1, f);
        std::fwrite(n.data(), 1, len, f);
    }
    return std::fclose(f) == 0;
}

// ---------------------------------------------------------------- checkpoint / resume
// Full training state: weights, biases, momentum buffers, epoch/step counters, best F1.
bool save_ckpt(const std::string& path, Net& net, int epochs_done, int step, double best_f1,
               int epochs_total, float lr0, int steps_per_epoch) {
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    int32_t hdr[8] = {0x4B434B52, 1, epochs_done, step, epochs_total, steps_per_epoch, int32_t(net.size()), IMG};
    std::fwrite(hdr, 4, 8, f);
    std::fwrite(&lr0, 4, 1, f);
    std::fwrite(&best_f1, 8, 1, f);
    for (Layer& l : net) {
        std::fwrite(l.w.data(), 4, l.w.size(), f);
        std::fwrite(l.b.data(), 4, l.b.size(), f);
        std::fwrite(l.vw.data(), 4, l.vw.size(), f);
        std::fwrite(l.vb.data(), 4, l.vb.size(), f);
    }
    if (std::fclose(f) != 0) return false;
    return std::rename(tmp.c_str(), path.c_str()) == 0;   // atomic replace
}

bool load_ckpt(const std::string& path, Net& net, int& epochs_done, int& step, double& best_f1,
               int epochs_total, float lr0, int steps_per_epoch) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::cerr << "No checkpoint " << path << "\n"; return false; }
    int32_t hdr[8];
    float lr = 0;
    double bf = 0;
    bool ok = std::fread(hdr, 4, 8, f) == 8 && std::fread(&lr, 4, 1, f) == 1 && std::fread(&bf, 8, 1, f) == 1;
    if (!ok || hdr[0] != 0x4B434B52 || hdr[6] != int32_t(net.size()) || hdr[7] != IMG) {
        std::cerr << "Checkpoint does not match this model/input size\n"; std::fclose(f); return false;
    }
    if (hdr[4] != epochs_total || hdr[5] != steps_per_epoch || std::fabs(lr - lr0) > 1e-9f) {
        std::cerr << "Checkpoint was made with epochs=" << hdr[4] << " lr=" << lr << " steps/epoch=" << hdr[5]
                  << " - use the same epochs, lr and dataset to resume (the LR schedule depends on them)\n";
        std::fclose(f); return false;
    }
    for (Layer& l : net) {
        if (std::fread(l.w.data(), 4, l.w.size(), f) != l.w.size() ||
            std::fread(l.b.data(), 4, l.b.size(), f) != l.b.size() ||
            std::fread(l.vw.data(), 4, l.vw.size(), f) != l.vw.size() ||
            std::fread(l.vb.data(), 4, l.vb.size(), f) != l.vb.size()) {
            std::cerr << "Checkpoint truncated\n"; std::fclose(f); return false;
        }
    }
    std::fclose(f);
    epochs_done = hdr[2]; step = hdr[3]; best_f1 = bf;
    return true;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : "dataset";
    int epochs = argc > 2 ? std::atoi(argv[2]) : 30;
    float lr0 = argc > 3 ? float(std::atof(argv[3])) : 0.01f;
    int limit = argc > 4 ? std::atoi(argv[4]) : 0;
    g_cpu_only = std::getenv("HYB_CPU") != nullptr;
    if (std::getenv("HYB_CHECK")) g_check = 5;
    if (g_cpu_only) std::cout << "*** CPU float-only mode (NPU bypassed) ***\n";

    std::vector<std::string> names = parse_names(root + "/data.yaml");
    int NC = int(names.size());
    if (NC < 1 || 5 + NC > 32) { std::cerr << "Could not parse class names from data.yaml (got " << NC << ")\n"; return 1; }
    std::cout << "Classes (" << NC << "):";
    for (auto& n : names) std::cout << " " << n;
    std::cout << "\n";

    std::vector<Sample> train, val;
    if (!load_split(root, "train", NC, limit, train) || !load_split(root, "val", NC, 0, val)) return 1;
    std::cout << "Loaded " << train.size() << " train / " << val.size() << " val images\n";

    // v2: stride-16 grid. Receptive field of an output cell is ~127 px (about 2/3 of the 192 px input),
    // so a cell can see most of a large object.
    const std::vector<Spec> specs = {
        {3, 16, 3, 2, 1, true},      // 192 -> 96
        {16, 32, 3, 2, 1, true},     //  96 -> 48
        {32, 64, 3, 2, 1, true},     //  48 -> 24
        {64, 128, 3, 2, 1, true},    //  24 -> 12
        {128, 128, 3, 1, 1, true},   //  12 -> 12
        {128, 128, 3, 1, 1, true},
        {128, 128, 3, 1, 1, true},
        {128, 5 + NC, 1, 1, 0, false}};
    Net net(specs.size());
    std::mt19937 rng(1234);
    int h = IMG, w = IMG;
    for (size_t i = 0; i < specs.size(); i++) {
        if (!layer_init(net[i], specs[i], h, w, i + 1 == specs.size(), rng)) return 1;
        h = net[i].hout; w = net[i].wout;
        std::cout << "L" << i << ": " << specs[i].cin << "->" << specs[i].cout << " k" << specs[i].k
                  << " s" << specs[i].s << "  out " << h << "x" << w << "  matmul K=" << net[i].Kp
                  << " N=" << net[i].Np << "\n";
    }
    if (h != S || w != S) { std::cerr << "Grid mismatch\n"; return 1; }

    const int steps_per_epoch = (int(train.size()) + BATCH - 1) / BATCH;
    const int total_steps = steps_per_epoch * epochs, warm = std::min(100, total_steps / 10 + 1);
    std::vector<int> idx(train.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    double best_f1 = -1;
    int step = 0;
    int start_ep = 0;
    if (std::getenv("HYB_RESUME")) {
        if (!load_ckpt("yolo_hybrid_ckpt.bin", net, start_ep, step, best_f1, epochs, lr0, steps_per_epoch)) return 1;
        std::cout << "Resumed from checkpoint: " << start_ep << " epoch(s) done, step " << step
                  << ", best F1 " << best_f1 << "\n";
        rng.seed(1234 + start_ep);   // fresh shuffle/augmentation sequence after the restart
    }
    std::vector<float> x, dhead;
    std::vector<Tgt> tg;

    for (int ep = start_ep; ep < epochs; ep++) {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(idx.begin(), idx.end(), rng);
        double eloss = 0; int seen = 0;
        for (int s0 = 0; s0 < int(idx.size()); s0 += BATCH, step++) {
            int bn = std::min<int>(BATCH, int(idx.size()) - s0);
            if (!quantize_all(net)) return 1;                 // push current weights to the NPU once per step
            for (Layer& l : net) {
                std::fill(l.gw.begin(), l.gw.end(), 0.f);
                std::fill(l.gb.begin(), l.gb.end(), 0.f);
            }
            for (int b = 0; b < bn; b++) {
                const Sample& s = train[idx[s0 + b]];
                std::vector<Box> ab;
                augment(s, rng, x, ab);
                if (!net_forward(net, x.data())) return 1;    // NPU
                build_targets(ab, false, tg);
                eloss += head_loss(net.back().out.data(), tg, NC, &dhead);
                net_backward(net, dhead);                     // CPU
            }
            seen += bn;

            float lr = step < warm ? lr0 * float(step + 1) / warm
                                   : lr0 * (0.05f + 0.95f * 0.5f * (1.f + std::cos(M_PI * double(step) / total_steps)));
            float inv = 1.f / bn;
            double sq = 0;
            for (Layer& l : net) {
                for (float g : l.gw) sq += double(g * inv) * (g * inv);
                for (float g : l.gb) sq += double(g * inv) * (g * inv);
            }
            float norm = float(std::sqrt(sq)), sc = inv * (norm > CLIP ? CLIP / norm : 1.f);
            for (Layer& l : net) {
                for (size_t i = 0; i < l.w.size(); i++) {
                    l.vw[i] = MOMENTUM * l.vw[i] - lr * (l.gw[i] * sc + WD * l.w[i]);
                    l.w[i] += l.vw[i];
                }
                for (size_t i = 0; i < l.b.size(); i++) {
                    l.vb[i] = MOMENTUM * l.vb[i] - lr * l.gb[i] * sc;
                    l.b[i] += l.vb[i];
                }
            }
            if ((step + 1) % 50 == 0)
                std::printf("  ep %d step %d  loss %.4f  gnorm %.2f  lr %.5f\n", ep + 1, step + 1,
                            eloss / seen, norm * 1.0f, lr);
        }
        EvalRes r;
        if (!evaluate(net, val, NC, r)) return 1;
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("Epoch %d/%d  train_loss %.4f  val_loss %.4f  P %.3f R %.3f F1 %.3f @IoU0.5 conf%.2f  (%.0fs)\n",
                    ep + 1, epochs, eloss / seen, r.loss, r.P, r.R, r.F1, EVAL_CONF, secs);
        if (!save_bin("yolo_hybrid_last.bin", net, names)) std::cerr << "Failed to write last.bin\n";
        if (r.F1 > best_f1) {
            best_f1 = r.F1;
            if (save_bin("yolo_hybrid_best.bin", net, names)) std::cout << "  saved yolo_hybrid_best.bin\n";
        }
        if (!save_ckpt("yolo_hybrid_ckpt.bin", net, ep + 1, step, best_f1, epochs, lr0, steps_per_epoch))
            std::cerr << "Failed to write checkpoint\n";
        if (fs::exists("STOP")) {
            std::cout << "STOP file found - checkpoint saved after epoch " << ep + 1
                      << ". Resume with: HYB_RESUME=1 (same arguments)\n";
            fs::remove("STOP");
            break;
        }
    }
    return 0;
}
