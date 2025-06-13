#include <cstdint>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <fstream>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using clk = std::chrono::high_resolution_clock;

// ============================================================
// 1. LcgPerm32  — mini‑PCG (LCG + xorshift + rotate) 32‑bit
// ============================================================
struct LcgPerm32 {
    u64 state;
    static constexpr u64 A = 6364136223846793005ULL;
    static constexpr u64 C = 1442695040888963407ULL;

    explicit LcgPerm32(u64 seed = 0) : state(seed) {}

    u32 next() {
        state = state * A + C;                                   // шаг LCG
        u32 x = static_cast<u32>(((state >> 18) ^ state) >> 27); // xorshift‑выжимка
        u32 rot = static_cast<u32>(state >> 59);                 // угол 0…31
        return (x >> rot) | (x << ((-rot) & 31));                // циклическая ротация
    }
};

// ============================================================
// 2. Xsw32 — XORShift64 + Weyl, вывод верхних 32 бит
// ============================================================
struct Xsw32 {
    u64 x, w;
    explicit Xsw32(u64 seed = 1) : x(seed ? seed : 1), w(~seed) {}

    u32 next() {
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        w += 0xb5ad4eceda1ce2a9ULL;       // Weyl‑шаг
        return static_cast<u32>((x + w) >> 32);
    }
};

// ============================================================
// 3. SaltedCb32 — SplitMix64 + двойная соль, средние 32 бит
// ============================================================
struct SaltedCb32 {
    u64 cnt, s1, s2;

    explicit SaltedCb32(u64 seed = 0)
        : cnt(seed),
          s1(0x9e3779b97f4a7c15ULL ^ seed),
          s2(0x60642e2a34326f15ULL + (seed << 1)) {}

    static u64 mix(u64 z, u64 a, u64 b) {
        z += a;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z ^= b;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    u32 next() { return static_cast<u32>(mix(++cnt, s1, s2) >> 16); } // средние 32 бита
};

// ============================================================
// 4. Mt32 - стандартный ГПСЧ std::mt19937 с методом next
// ============================================================
struct Mt32 {
    std::mt19937 e;
    explicit Mt32(u32 seed) { e.seed(seed); }
    u32 next() { return e(); }     // теперь интерфейс такой же, как у других ГПСЧ
};

// ============================================================
//  Вспомогательные функции: gammaQ (верхняя регуляриз. гамма)
// ============================================================
namespace impl {
    double gamma_series(double a, double x) {
        const int ITMAX = 1000; const double EPS = 1e-9;
        double sum = 1.0 / a, term = sum;
        for (int n = 1; n <= ITMAX; ++n) {
            term *= x / (a + n);
            sum += term;
            if (fabs(term) < fabs(sum) * EPS) break;
        }
        return sum * std::exp(-x + a * std::log(x));
    }

    double gamma_cf(double a, double x) {
        const int ITMAX = 1000; const double EPS = 1e-9; const double FPMIN = 1e-30;
        double b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, f = d;
        for (int i = 1; i <= ITMAX; ++i) {
            double an = -i * (i - a);
            b += 2; d = an * d + b; if (fabs(d) < FPMIN) d = FPMIN;
            c = b + an / c;        if (fabs(c) < FPMIN) c = FPMIN;
            d = 1 / d; double delta = d * c; f *= delta;
            if (fabs(delta - 1) < EPS) break;
        }
        return std::exp(-x + a * std::log(x)) * f;
    }
}

double gammaQ(double a, double x) {
    if (x < 0 || a <= 0) return 0;
    if (x == 0) return 1;
    if (x < a + 1) {
        double P = impl::gamma_series(a, x) / std::tgamma(a);
        return 1 - P;
    }
    return impl::gamma_cf(a, x) / std::tgamma(a);
}

inline double Phi(double x) { return 0.5 * (1 + std::erf(x / M_SQRT2)); }

// ============================================================
//  Реализация 5 базовых тестов NIST‑STS (p‑value)
// ============================================================
double p_monobit(const std::vector<int>& bits) {
    long S = 0; 
    for (int b : bits) S += b ? 1 : -1;
    double sobs = std::fabs(S) / std::sqrt(bits.size());
    return std::erfc(sobs / M_SQRT2);
}

double p_blockfreq(const std::vector<int>& bits, int M = 128) {
    int n = bits.size(); 
    int N = n / M; 
    if (N == 0) return 0;
    double chi = 0;
    for (int i = 0; i < N; ++i) {
        int ones = 0; for (int j = 0; j < M; ++j) ones += bits[i * M + j];
        double pi = double(ones) / M;
        chi += (pi - 0.5) * (pi - 0.5);
    }
    return gammaQ(N / 2.0, 4 * M * chi / 2.0);
}

double p_runs(const std::vector<int>& bits) {
    int n = bits.size(), ones = 0; 
    for (int b : bits) ones += b;
    double pi = double(ones) / n;
    if (std::fabs(pi - 0.5) > (2.0 / std::sqrt(n))) return 0;
    int runs = 1; for (int i = 0; i < n - 1; ++i) if (bits[i] != bits[i + 1]) ++runs;
    double expR = 2 * n * pi * (1 - pi);
    double var  = 2 * M_SQRT2 * std::sqrt(n) * pi * (1 - pi);
    double z = std::fabs(runs - expR) / var;
    return std::erfc(z);
}

double p_longest_run(const std::vector<int>& bits) {
    int n = bits.size(); 
    if (n < 128) return 0;
    int M = (n < 6272) ? 8 : (n < 750000 ? 128 : 10000);
    int N = n / M; 
    if (N <= 0) return 0;

    int K; std::vector<int> v; std::vector<double> P;
    if (M == 8) {
        K = 3; v.assign(4, 0); P = {0.2148, 0.3672, 0.2305, 0.1875};
    } else if (M == 128) {
        K = 5; v.assign(6, 0); P = {0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124};
    } else {
        K = 6; v.assign(7, 0); P = {0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727};
    }

    for (int i = 0; i < N; ++i) {
        int maxRun = 0;
        int currentRun = 0;
        for (int j = 0; j < M; ++j) {
            if (bits[i * M + j] == 1) {
                currentRun++;
                if (currentRun > maxRun) {
                    maxRun = currentRun;
                }
            } else {
                currentRun = 0;
            }
        }
        if (M == 8) {
            if (maxRun <= 1) v[0]++;
            else if (maxRun == 2) v[1]++;
            else if (maxRun == 3) v[2]++;
            else if (maxRun >= 4) v[3]++;
        } else if (M == 128) {
            if (maxRun <= 4) v[0]++;
            else if (maxRun == 5) v[1]++;
            else if (maxRun == 6) v[2]++;
            else if (maxRun == 7) v[3]++;
            else if (maxRun == 8) v[4]++;
            else if (maxRun >= 9) v[5]++;
        } else if (M == 10000) {
            if (maxRun <= 10) v[0]++;
            else if (maxRun == 11) v[1]++;
            else if (maxRun == 12) v[2]++;
            else if (maxRun == 13) v[3]++;
            else if (maxRun == 14) v[4]++;
            else if (maxRun == 15) v[5]++;
            else if (maxRun >= 16) v[6]++;
        }
    }

    double chi = 0;
    for (size_t i = 0; i < K + 1; ++i) {
        double exp = N * P[i]; 
        double diff = v[i] - exp;
        chi += diff * diff / exp;
    }
    return gammaQ(K / 2.0, chi / 2.0);
}

double p_cusum(const std::vector<int>& bits) {
    int n = bits.size();
    long S = 0, maxAbs = 0;
    for (int b : bits) { S += b ? 1 : -1; maxAbs = std::max(maxAbs, std::labs(S)); }
    double z = maxAbs / std::sqrt(n), p = 1.0;
    for (int k = -(n / 4) + 1; k <= n / 4; ++k)
        p -= Phi((4 * k + 1) * z) - Phi((4 * k - 1) * z);
    for (int k = -(n / 4) + 1; k <= n / 4; ++k)
        p += Phi((4 * k + 3) * z) - Phi((4 * k + 1) * z);
    return p;
}

// ============================================================
//  χ²‑тест равномерности (10 корзин, df = 9)
// ============================================================

double chi_square(const std::vector<u32>& v) {
    const int BINS = 10;           // число корзин по правилу Стерджеса
    std::size_t n = v.size();
    double expected = static_cast<double>(n) / BINS;
    std::uint64_t mask = 0xFFFFFFFFULL; // 32‑бит макс

    std::array<int, BINS> freq{};
    for (u32 val : v) {
        // Быстрый способ: (val * BINS) >> 32 даёт 0..BINS‑1
        int idx = static_cast<int>((static_cast<u64>(val) * BINS) >> 32);
        ++freq[idx];
    }

    double chi = 0.0;
    for (int k = 0; k < BINS; ++k) {
        double diff = freq[k] - expected;
        chi += diff * diff / expected;
    }
    return chi; // df = 9
}

// ------------------------------------------------------------
//  Тестирование одного генератора (20 выборок × 1000)
// ------------------------------------------------------------

template <typename Gen>
void test_generator(const std::string& tag, Gen& rng) {
    constexpr int SAMPLES = 20, SIZE = 1000;
    constexpr double CHI_LOW = 5.899, CHI_HIGH = 11.389; // p‑коридор 25–75 %

    std::cout << "\n=== " << tag << " ===\n";
    for (int s = 0; s < SAMPLES; ++s) {
        std::vector<u32> vals; 
        vals.reserve(SIZE);
        long double sum = 0, sum2 = 0;
        for (int i = 0; i < SIZE; ++i) {
            u32 val = rng.next();
            vals.push_back(val);
            sum += val;
            sum2 += (long double)val*val;
        }

        double mean = double(sum) / SIZE;
        double var  = double(sum2)/SIZE - mean*mean;
        double sigma = std::sqrt(var);
        double cv = sigma / mean;

        // χ² равномерности
        double chi = chi_square(vals);
        bool chi_ok = (chi >= CHI_LOW && chi <= CHI_HIGH);

        // битовый поток
        std::vector<int> bits; bits.reserve(SIZE * 32);
        for (u32 v : vals)
            for (int b = 31; b >= 0; --b) bits.push_back((v >> b) & 1);

        double p1 = p_monobit(bits);
        double p2 = p_blockfreq(bits);
        double p3 = p_runs(bits);
        double p4 = p_longest_run(bits);
        double p5 = p_cusum(bits);

        auto ok = [&](double p) { return p > 0.01; };

        std::cout << std::fixed << std::setprecision(2)
                  << "#" << std::setw(2) << s
                  << " mean=" << mean << std::setprecision(2)
                  << " σ=" << sigma
                  << " CV=" << cv << std::setprecision(3)
                  << " χ²=" << std::setw(6) << chi << (chi_ok ? " " : "*")
                  << " p1=" << std::setw(5) << p1 << (ok(p1) ? " " : "*")
                  << " p2=" << std::setw(5) << p2 << (ok(p2) ? " " : "*")
                  << " p3=" << std::setw(5) << p3 << (ok(p3) ? " " : "*")
                  << " p4=" << std::setw(5) << p4 << (ok(p4) ? " " : "*")
                  << " p5=" << std::setw(5) << p5 << (ok(p5) ? " " : "*")
                  << '\n';
    }
}

// ============================================================
//  Измерение времени генерации N чисел
// ============================================================

template<typename Gen>
long long bench(Gen& rng, int N){
    auto t0 = clk::now();
    for(int i=0;i<N;++i) rng.next();
    auto t1 = clk::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
}

void benchmark_speed(){
    const std::vector<int> sizes = {1'000, 25'000, 50'000, 100'000, 200'000, 350'000, 500'000, 750'000, 1'000'000};
    std::ofstream csv("rng_speed.csv");
    csv << "N,LCG(ns),XSW(ns),Salted(ns),mt19937(ns)\n";

    for(int N : sizes){
        LcgPerm32  g1(123456ULL); Xsw32 g2(123456ULL); SaltedCb32 g3(123456ULL); Mt32 mt(123456ULL);
        long long t1 = bench(g1,N);
        long long t2 = bench(g2,N);
        long long t3 = bench(g3,N);
        long long t4 = bench(mt,N);
        csv << N << ',' << t1 << ',' << t2 << ',' << t3 << ',' << t4 << "\n";
    }
    csv.close();
    std::cout << "Тайминги сохранены в rng_speed.csv (наносекунд).\n";
}

// ============================================================
//                      main()
// ============================================================

int main() {
    LcgPerm32  g1(123456ULL);
    Xsw32      g2(123456ULL);
    SaltedCb32 g3(123456ULL);

    test_generator("LcgPerm32",  g1);
    test_generator("Xsw32",      g2);
    test_generator("SaltedCb32", g3);

    benchmark_speed();
    return 0;
}
