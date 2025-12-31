// sa_runner.cpp
// 编译示例：g++ -O3 -std=c++17 -fopenmp -march=native sa_runner.cpp -o sa_runner.exe
// 功能：对每个分组执行模拟退火以最小化包络边长，输入/输出为 CSV。
// 复杂度概览：单组退火约 O(iters * n)（NV=15 视为常数，碰撞 O(n)）；全部分组为其累加
// .\sa_runner.exe --dimer "C:\kaggle\70.916_sa.csv" "C:\kaggle\output.csv" 1 8 200
#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

using ld = long double; // 使用长双精度做几何计算

struct Vec { ld x, y; }; // 简单 2D 向量
static inline Vec operator+(const Vec& a, const Vec& b){ return {a.x+b.x, a.y+b.y}; }
static inline Vec operator-(const Vec& a, const Vec& b){ return {a.x-b.x, a.y-b.y}; }
static inline ld cross(const Vec& a, const Vec& b){ return a.x*b.y - a.y*b.x; }

struct Bounds { ld minx, miny, maxx, maxy; }; // AABB 边界盒

static constexpr int NV = 15;       // 多边形顶点数（固定常数）
static constexpr ld EPS = 1e-12L;   // 几何容差

// 64 位可重复随机数生成器
struct SplitMix64 {
    uint64_t x;
    explicit SplitMix64(uint64_t seed=0x123456789ULL) : x(seed) {}
    uint64_t next_u64(){
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    ld next01(){ return (ld)((next_u64() >> 11) * (1.0L / (ld)(1ULL<<53))); } // [0,1)
    int randint(int lo, int hi){ return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
};

// 带容差符号
static inline int sgn(ld v){
    if (v > EPS) return 1;
    if (v < -EPS) return -1;
    return 0;
}

// 判断点p是否在线段ab上（含容差）
static inline bool on_segment(const Vec& a, const Vec& b, const Vec& p){
    if (fabsl(cross(b-a, p-a)) > EPS) return false;
    return (min(a.x,b.x) - EPS <= p.x && p.x <= max(a.x,b.x) + EPS &&
            min(a.y,b.y) - EPS <= p.y && p.y <= max(a.y,b.y) + EPS);
}

// 严格相交（排除共线接触）
static inline bool proper_intersect(const Vec& a1, const Vec& a2, const Vec& b1, const Vec& b2){
    Vec A = a2 - a1;
    Vec B = b2 - b1;
    ld c1 = cross(A, b1 - a1);
    ld c2 = cross(A, b2 - a1);
    ld c3 = cross(B, a1 - b1);
    ld c4 = cross(B, a2 - b1);
    int o1 = sgn(c1), o2 = sgn(c2), o3 = sgn(c3), o4 = sgn(c4);
    return (o1*o2 < 0) && (o3*o4 < 0);
}

// 点在多边形中的位置：1 内部，0 边界，-1 外部
static int point_in_poly_strict(const Vec& p, const array<Vec,NV>& poly){
    int cnt = 0;
    for(int i=0;i<NV;i++){
        Vec a = poly[i];
        Vec b = poly[(i+1)%NV];
        if (on_segment(a,b,p)) return 0;

        bool up = (a.y <= p.y + EPS) && (b.y >  p.y + EPS);
        bool dn = (b.y <= p.y + EPS) && (a.y >  p.y + EPS);
        if (up || dn){
            ld xint = a.x + (b.x - a.x) * ((p.y - a.y) / (b.y - a.y));
            if (xint > p.x + EPS) cnt ^= 1;
        }
    }
    return cnt ? 1 : -1;
}

// AABB 剪枝：判断是否分离
static inline bool aabb_disjoint(const Bounds& A, const Bounds& B){
    return (A.maxx < B.minx || A.minx > B.maxx || A.maxy < B.miny || A.miny > B.maxy);
}

static array<Vec,NV> BASE_V;    // 基础树形顶点
static Vec BASE_CENTROID;       // 基础多边形质心

// 计算基础多边形质心，O(NV)
static void compute_base_centroid(){
    ld A = 0, Cx = 0, Cy = 0;
    for(int i=0;i<NV;i++){
        const Vec& p = BASE_V[i];
        const Vec& q = BASE_V[(i+1)%NV];
        ld cr = cross(p, q);
        A += cr;
        Cx += (p.x + q.x) * cr;
        Cy += (p.y + q.y) * cr;
    }
    A *= 0.5L;
    if (fabsl(A) < EPS){
        ld sx=0, sy=0;
        for(auto &p: BASE_V){ sx += p.x; sy += p.y; }
        BASE_CENTROID = {sx/NV, sy/NV};
        return;
    }
    Cx /= (6.0L * A);
    Cy /= (6.0L * A);
    BASE_CENTROID = {Cx, Cy};
}

static inline ld deg2rad(ld d){ return d * acosl(-1.0L) / 180.0L; }

// 单棵树的状态
struct Tree {
    ld cx, cy, ang_deg;
    array<Vec,NV> v;
    Bounds b;
    Vec centroid;
};

// 根据中心/角度生成树的多边形与边界，O(NV)
static Tree build_tree(ld cx, ld cy, ld ang_deg){
    Tree t;
    t.cx = cx; t.cy = cy; t.ang_deg = ang_deg;
    ld r = deg2rad(ang_deg);
    ld c = cosl(r), s = sinl(r);

    Bounds bb;
    bb.minx = bb.miny = numeric_limits<ld>::infinity();
    bb.maxx = bb.maxy = -numeric_limits<ld>::infinity();

    for(int i=0;i<NV;i++){
        ld x = BASE_V[i].x, y = BASE_V[i].y;
        ld nx = x*c - y*s + cx;
        ld ny = x*s + y*c + cy;
        t.v[i] = {nx, ny};
        bb.minx = min(bb.minx, nx);
        bb.miny = min(bb.miny, ny);
        bb.maxx = max(bb.maxx, nx);
        bb.maxy = max(bb.maxy, ny);
    }
    t.b = bb;

    { // centroid
        ld x = BASE_CENTROID.x, y = BASE_CENTROID.y;
        t.centroid = {x*c - y*s + cx, x*s + y*c + cy};
    }
    return t;
}

// ---- 小工具：旋转/包络 ----
static inline Vec rot_deg(const Vec& v, ld deg){
    ld r = deg2rad(deg);
    ld c = cosl(r), s = sinl(r);
    return {v.x*c - v.y*s, v.x*s + v.y*c};
}
static inline Bounds bounds_union(const Bounds& a, const Bounds& b){
    return {
        min(a.minx, b.minx),
        min(a.miny, b.miny),
        max(a.maxx, b.maxx),
        max(a.maxy, b.maxy)
    };
}

// ---- 2聚体（解析几何定义，不做试探）----
// 约定：局部坐标系下，A 树中心在 (0,0)，B 树中心在 d0；整体再绕 A 中心旋转 phi。
// 你的相切条件（对应 sa_runner.cpp 基础树形顶点）：
// - 树冠尖端：p_tip=(0,0.8) (index0)
// - “第三层(最顶层)端点”：p_l3=(0.125,0.5) (index1)
// 基准朝向：A 向右(270°)，B 向左(90°)。
// 相切方程：CA + R(270)p_tip = CB + R(90)p_l3
// 得 d0 = CB-CA = R(270)p_tip - R(90)p_l3 = (1.3, -0.125)
// 为保证“恰好不严格相交”，在 x 方向加 1e-12 余量：d0.x += 1e-12
struct Dimer {
    static constexpr ld ANG_A0 = 270.0L;
    static constexpr ld ANG_B0 = 90.0L;
    static inline Vec d0(){
        return {1.3L + 1e-12L, -0.125L};
    }

    // A锚点中心为 (ax,ay)，整体旋转 phi_deg，输出两棵树
    static inline array<Tree,2> build(ld ax, ld ay, ld phi_deg){
        Vec d = rot_deg(d0(), phi_deg);
        ld a_ang = ANG_A0 + phi_deg;
        ld b_ang = ANG_B0 + phi_deg;
        Tree A = build_tree(ax, ay, a_ang);
        Tree B = build_tree(ax + d.x, ay + d.y, b_ang);
        return {A, B};
    }

    // 局部包络：A 在(0,0)，整体旋转 phi 后的 2棵树包络
    static inline Bounds local_bounds(ld phi_deg){
        auto ts = build(0.0L, 0.0L, phi_deg);
        return bounds_union(ts[0].b, ts[1].b);
    }
};

// 严格重叠检测（先 AABB，再边交/点内），O(NV^2)
static bool strict_overlap_candidate(const Tree& A, const Tree& B){
    if (aabb_disjoint(A.b, B.b)) return false;

    for(int i=0;i<NV;i++){
        Vec a1 = A.v[i];
        Vec a2 = A.v[(i+1)%NV];
        for(int j=0;j<NV;j++){
            Vec b1 = B.v[j];
            Vec b2 = B.v[(j+1)%NV];
            if (proper_intersect(a1,a2,b1,b2)) return true;
        }
    }
    for(int i=0;i<NV;i++) if (point_in_poly_strict(A.v[i], B.v) == 1) return true;
    for(int j=0;j<NV;j++) if (point_in_poly_strict(B.v[j], A.v) == 1) return true;

    if (point_in_poly_strict(A.centroid, B.v) == 1) return true;
    if (point_in_poly_strict(B.centroid, A.v) == 1) return true;

    return false;
}

// 聚合所有 AABB 得到总包络，O(n)
static inline Bounds envelope_from_bounds(const vector<Bounds>& bs){
    if(bs.empty()) return {0,0,0,0};
    Bounds e = bs[0];
    for(size_t i=1;i<bs.size();i++){
        e.minx = min(e.minx, bs[i].minx);
        e.miny = min(e.miny, bs[i].miny);
        e.maxx = max(e.maxx, bs[i].maxx);
        e.maxy = max(e.maxy, bs[i].maxy);
    }
    return e;
}
// 替换一个 AABB 后重算总包络，O(n)
static inline Bounds envelope_replace(const vector<Bounds>& bs, int rep, const Bounds& nb){
    Bounds e = (rep==0? nb : bs[0]);
    for(int i=1;i<(int)bs.size();i++){
        const Bounds& b = (i==rep? nb : bs[i]);
        e.minx = min(e.minx, b.minx);
        e.miny = min(e.miny, b.miny);
        e.maxx = max(e.maxx, b.maxx);
        e.maxy = max(e.maxy, b.maxy);
    }
    return e;
}
// 侧长（正方形边长）= max(宽, 高)
static inline ld side_len(const Bounds& e){
    return max(e.maxx - e.minx, e.maxy - e.miny);
}

// 退火返回的结果统计
struct SAResult {
    vector<Tree> best;
    ld best_side;
    long long iters=0, accepted=0, collided=0;
};

// 对单个分组执行模拟退火
// 复杂度：iters * O(n)（碰撞与包络更新，NV 为常数）；iters 为 max_iterations 或其 3 倍
static SAResult run_sa_group(
    const vector<Tree>& initial,
    int max_iterations,
    ld t_start,
    ld t_end,
    SplitMix64& rng,
    double per_group_time_sec,
    int verbose
){
    int n = (int)initial.size();
    bool small = (n <= 50); // 小组与大组采用不同参数

    int iters = small ? max_iterations * 3 : max_iterations; // 小组更多迭代
    ld T0 = small ? t_start * 2.0L : t_start;                // 小组初温更高
    ld gravity_w = small ? 1e-4L : 1e-6L;                    // 重力惩罚
    // ld gravity_w = 0.0L;

    vector<Tree> cur = initial;
    vector<Bounds> cur_bounds(n);
    for(int i=0;i<n;i++) cur_bounds[i] = cur[i].b;

    Bounds env = envelope_from_bounds(cur_bounds); // 初始包络

    ld dist_sum = 0;
    for(int i=0;i<n;i++) dist_sum += cur[i].cx*cur[i].cx + cur[i].cy*cur[i].cy;

    // 簇类 move 概率（大组更需要协同动作）
    // - cluster2: 两棵树以“公共质心”(两中心中点)做点对称镜像后，再整体旋转一个角度
    // - cluster3: 三棵相邻树同向平移（不旋转）
    const ld cluster2_prob = (n >= 2 ? (small ? 0.10L : 0.22L) : 0.0L);
    const ld cluster3_prob = (n >= 3 ? (small ? 0.06L : 0.14L) : 0.0L);

    // 能量：侧长 + 重力项，返回 (E, side)
    auto energy = [&](const Bounds& e, ld dsum)->pair<ld, ld>{
        ld s = side_len(e);
        ld norm_d = dsum / (ld)max(1,n);
        ld E = s + gravity_w * norm_d;
        return {E, s};
    };

    auto [curE, curS] = energy(env, dist_sum); // 当前能量/侧长

    vector<Tree> best = cur; // 记录当前最优解
    ld bestS = curS;

    ld cool = powl(t_end / T0, 1.0L / (ld)iters); // 指数降温因子
    ld T = T0;                                   // 当前温度

    auto t0 = chrono::steady_clock::now();

    long long accepted_cnt=0, collided_cnt=0;
    for(int it=0; it<iters; it++){
        if(per_group_time_sec > 0){ // 单组时间限制
            double el = chrono::duration<double>(chrono::steady_clock::now()-t0).count();
            if(el > per_group_time_sec) break;
        }
        if(verbose && (it % 200000 == 0)){ // 可选迭代日志
            cout << "[sa] it=" << it << "/" << iters
                 << " T=" << (double)T
                 << " curS=" << (double)curS
                 << " bestS=" << (double)bestS
                 << " acc=" << accepted_cnt
                 << " col=" << collided_cnt
                 << "\n" << flush;
        }

        ld progress = (ld)it / (ld)iters; // 0~1 进度

        ld move_scale, rot_scale;
        if(small){
            move_scale = max((ld)0.005L, (ld)3.0L * (1.0L - progress));
            rot_scale  = max((ld)0.001L, (ld)5.0L * (1.0L - progress));
        }else{
            move_scale = max((ld)0.001L, (ld)1.0L * (T / T0));
            rot_scale  = max((ld)0.002L, (ld)5.0L * (T / T0));
        }

        // --- Move selection: 2-tree 簇镜像+整体旋转 / 3-tree 簇平移 / 单树随机扰动 ---
        ld rsel = rng.next01();

        // (A) 2-tree 簇：以两棵树中心的中点为“公共质心”，先点对称镜像，再整体旋转一个角度
        if(cluster2_prob > 0.0L && rsel < cluster2_prob){
            // 改为遍历所有树作为 i0（而不是随机挑一个）
            // 策略：按循环顺序扫一遍，找到“第一个不碰撞”的候选簇（i0, nearest(i0)）就执行一次 Metropolis
            int start = rng.randint(0, n-1);

            // 整体旋转角（度），每轮 cluster2 固定一次，避免在遍历中引入过多随机性与开销
            ld dphi = (rng.next01() - 0.5L) * (rot_scale * 1.5L);
            ld rr = deg2rad(dphi);
            ld c = cosl(rr), s = sinl(rr);

            bool found = false;
            array<int,2> chosen = {-1, -1};
            array<Tree,2> chosenT;
            array<Bounds,2> chosenB;

            for(int step=0; step<n; step++){
                int i0 = (start + step) % n;
                int i1 = -1;
                ld bestd = numeric_limits<ld>::infinity();
                for(int j=0;j<n;j++){
                    if(j==i0) continue;
                    ld dx0 = cur[j].cx - cur[i0].cx;
                    ld dy0 = cur[j].cy - cur[i0].cy;
                    ld d2  = dx0*dx0 + dy0*dy0;
                    if(d2 < bestd){ bestd = d2; i1 = j; }
                }
                if(i1 < 0) continue;

                array<int,2> idxs = {i0, i1};
                auto in_cluster = [&](int k)->bool{ return (k==idxs[0] || k==idxs[1]); };

                ld Cx = (cur[idxs[0]].cx + cur[idxs[1]].cx) * 0.5L;
                ld Cy = (cur[idxs[0]].cy + cur[idxs[1]].cy) * 0.5L;

                array<Tree,2> candT;
                array<Bounds,2> candB;
                for(int t=0;t<2;t++){
                    const Tree& o = cur[idxs[t]];
                    // 先点对称镜像：r -> -r（关于 C 的中心对称）
                    ld rx = o.cx - Cx;
                    ld ry = o.cy - Cy;
                    rx = -rx; ry = -ry;
                    // 再整体旋转 dphi：r -> R(dphi) r
                    ld nrx = rx*c - ry*s;
                    ld nry = rx*s + ry*c;
                    ld ncx = Cx + nrx;
                    ld ncy = Cy + nry;

                    // 角度：镜像（中心对称）相当于整体 180° 翻转，再加整体旋转 dphi
                    ld nang = o.ang_deg + 180.0L + dphi;
                    nang = fmodl(nang, 360.0L);
                    if(nang < 0) nang += 360.0L;

                    candT[t] = build_tree(ncx, ncy, nang);
                    candB[t] = candT[t].b;
                }

                bool collision = false;
                for(int t=0;t<2 && !collision;t++){
                    for(int k=0;k<n;k++){
                        if(in_cluster(k)) continue;
                        if(aabb_disjoint(candB[t], cur_bounds[k])) continue;
                        if(strict_overlap_candidate(candT[t], cur[k])){
                            collision = true;
                            break;
                        }
                    }
                }
                if(!collision){
                    found = true;
                    chosen = idxs;
                    chosenT = candT;
                    chosenB = candB;
                    break;
                }
            }

            if(!found){
                // 没有找到可行的 2-tree 候选簇，本轮拒绝并降温
                collided_cnt++;
                T *= cool;
                continue;
            }

            auto in_cluster = [&](int k)->bool{ return (k==chosen[0] || k==chosen[1]); };

            // 更新重力距离平方和（2 个点更新）
            ld old_d = 0, new_d = 0;
            for(int t=0;t<2;t++){
                const Tree& o = cur[chosen[t]];
                old_d += o.cx*o.cx + o.cy*o.cy;
                new_d += chosenT[t].cx*chosenT[t].cx + chosenT[t].cy*chosenT[t].cy;
            }
            ld cand_dsum = dist_sum - old_d + new_d;

            // 包络：O(n) 直接扫描替换两个 bounds
            Bounds cand_env;
            cand_env.minx = cand_env.miny = numeric_limits<ld>::infinity();
            cand_env.maxx = cand_env.maxy = -numeric_limits<ld>::infinity();
            for(int k=0;k<n;k++){
                const Bounds* bptr = &cur_bounds[k];
                if(k==chosen[0]) bptr = &chosenB[0];
                else if(k==chosen[1]) bptr = &chosenB[1];
                const Bounds& b = *bptr;
                cand_env.minx = min(cand_env.minx, b.minx);
                cand_env.miny = min(cand_env.miny, b.miny);
                cand_env.maxx = max(cand_env.maxx, b.maxx);
                cand_env.maxy = max(cand_env.maxy, b.maxy);
            }

            auto [newE, newS] = energy(cand_env, cand_dsum);
            ld delta = newE - curE;

            bool accept = false;
            if(delta < 0){
                accept = true;
            }else if(T > 1e-12L){
                ld prob = expl(-delta * 1000.0L / T);
                accept = (rng.next01() < prob);
            }

            if(accept){
                accepted_cnt++;
                for(int t=0;t<2;t++){
                    cur[chosen[t]] = chosenT[t];
                    cur_bounds[chosen[t]] = chosenB[t];
                }
                env = cand_env;
                dist_sum = cand_dsum;
                curE = newE;
                curS = newS;
                if(newS < bestS){
                    bestS = newS;
                    best = cur;
                }
            }
            T *= cool;
            continue;
        }

        // (B) 3-tree 簇平移：选一棵 + 两个最近邻，同向平移（不旋转）
        if(cluster3_prob > 0.0L && rsel < (cluster2_prob + cluster3_prob)){
            int i0 = rng.randint(0, n-1);
            int i1 = -1, i2 = -1;
            ld bestd1 = numeric_limits<ld>::infinity();
            ld bestd2 = numeric_limits<ld>::infinity();
            for(int j=0;j<n;j++){
                if(j==i0) continue;
                ld dx0 = cur[j].cx - cur[i0].cx;
                ld dy0 = cur[j].cy - cur[i0].cy;
                ld d2  = dx0*dx0 + dy0*dy0;
                if(d2 < bestd1){
                    bestd2 = bestd1; i2 = i1;
                    bestd1 = d2;    i1 = j;
                }else if(d2 < bestd2){
                    bestd2 = d2;    i2 = j;
                }
            }
            if(i1 >= 0 && i2 >= 0){
                array<int,3> idxs = {i0, i1, i2};
                auto in_cluster = [&](int k)->bool{ return (k==idxs[0] || k==idxs[1] || k==idxs[2]); };

                ld dx = (rng.next01() - 0.5L) * 0.08L * move_scale;
                ld dy = (rng.next01() - 0.5L) * 0.08L * move_scale;

                array<Tree,3> candT;
                array<Bounds,3> candB;
                for(int t=0;t<3;t++){
                    const Tree& o = cur[idxs[t]];
                    ld ncx = o.cx + dx;
                    ld ncy = o.cy + dy;
                    candT[t] = build_tree(ncx, ncy, o.ang_deg); // 簇平移：角度不变
                    candB[t] = candT[t].b;
                }

                bool collision = false;
                for(int t=0;t<3 && !collision;t++){
                    for(int k=0;k<n;k++){
                        if(in_cluster(k)) continue;
                        if(aabb_disjoint(candB[t], cur_bounds[k])) continue;
                        if(strict_overlap_candidate(candT[t], cur[k])){
                            collision = true;
                            break;
                        }
                    }
                }
                if(collision){
                    collided_cnt++;
                    T *= cool;
                    continue;
                }

                ld old_d = 0, new_d = 0;
                for(int t=0;t<3;t++){
                    const Tree& o = cur[idxs[t]];
                    old_d += o.cx*o.cx + o.cy*o.cy;
                    new_d += candT[t].cx*candT[t].cx + candT[t].cy*candT[t].cy;
                }
                ld cand_dsum = dist_sum - old_d + new_d;

                Bounds cand_env;
                cand_env.minx = cand_env.miny = numeric_limits<ld>::infinity();
                cand_env.maxx = cand_env.maxy = -numeric_limits<ld>::infinity();
                for(int k=0;k<n;k++){
                    const Bounds* bptr = &cur_bounds[k];
                    if(k==idxs[0]) bptr = &candB[0];
                    else if(k==idxs[1]) bptr = &candB[1];
                    else if(k==idxs[2]) bptr = &candB[2];
                    const Bounds& b = *bptr;
                    cand_env.minx = min(cand_env.minx, b.minx);
                    cand_env.miny = min(cand_env.miny, b.miny);
                    cand_env.maxx = max(cand_env.maxx, b.maxx);
                    cand_env.maxy = max(cand_env.maxy, b.maxy);
                }

                auto [newE, newS] = energy(cand_env, cand_dsum);
                ld delta = newE - curE;

                bool accept = false;
                if(delta < 0){
                    accept = true;
                }else if(T > 1e-12L){
                    ld prob = expl(-delta * 1000.0L / T);
                    accept = (rng.next01() < prob);
                }

                if(accept){
                    accepted_cnt++;
                    for(int t=0;t<3;t++){
                        cur[idxs[t]] = candT[t];
                        cur_bounds[idxs[t]] = candB[t];
                    }
                    env = cand_env;
                    dist_sum = cand_dsum;
                    curE = newE;
                    curS = newS;
                    if(newS < bestS){
                        bestS = newS;
                        best = cur;
                    }
                }
                T *= cool;
                continue;
            }
            // 若邻居不足则退化到单树
        }

        // (C) 单树随机平移/旋转扰动（原逻辑）
        int idx = rng.randint(0, n-1);
        const Tree& orig = cur[idx];
        const Bounds& ob = cur_bounds[idx];

        ld dx = (rng.next01() - 0.5L) * 0.1L * move_scale;
        ld dy = (rng.next01() - 0.5L) * 0.1L * move_scale;
        ld dang = (rng.next01() - 0.5L) * rot_scale;

        ld ncx = orig.cx + dx;
        ld ncy = orig.cy + dy;
        ld nang = orig.ang_deg + dang;

        Tree cand = build_tree(ncx, ncy, nang); // 构造候选

        bool collision = false; // 碰撞检测 O(n)
        for(int k=0;k<n;k++){
            if(k==idx) continue;
            if(aabb_disjoint(cand.b, cur_bounds[k])) continue;
            if(strict_overlap_candidate(cand, cur[k])){
                collision = true;
                break;
            }
        }
        if(collision){ // 碰撞则拒绝并降温
            collided_cnt++;
            T *= cool;
            continue;
        }

        // 更新重力距离平方和
        ld old_d = orig.cx*orig.cx + orig.cy*orig.cy;
        ld new_d = ncx*ncx + ncy*ncy;
        ld cand_dsum = dist_sum - old_d + new_d;

        // 是否破坏包络极值，决定全量/增量更新
        bool need_recompute = (
            (fabsl(ob.minx - env.minx) < EPS && cand.b.minx > env.minx + EPS) ||
            (fabsl(ob.miny - env.miny) < EPS && cand.b.miny > env.miny + EPS) ||
            (fabsl(ob.maxx - env.maxx) < EPS && cand.b.maxx < env.maxx - EPS) ||
            (fabsl(ob.maxy - env.maxy) < EPS && cand.b.maxy < env.maxy - EPS)
        );

        Bounds cand_env;
        if(need_recompute){
            cand_env = envelope_replace(cur_bounds, idx, cand.b);
        }else{
            cand_env = env;
            cand_env.minx = min(cand_env.minx, cand.b.minx);
            cand_env.miny = min(cand_env.miny, cand.b.miny);
            cand_env.maxx = max(cand_env.maxx, cand.b.maxx);
            cand_env.maxy = max(cand_env.maxy, cand.b.maxy);
        }

        auto [newE, newS] = energy(cand_env, cand_dsum); // 计算新能量
        ld delta = newE - curE;                          // 能量差

        bool accept = false; // Metropolis 接受判断
        if(delta < 0){
            accept = true;
        }else if(T > 1e-12L){
            ld prob = expl(-delta * 1000.0L / T);
            accept = (rng.next01() < prob);
        }

        if(accept){ // 接受则提交状态
            accepted_cnt++;
            cur[idx] = cand;
            cur_bounds[idx] = cand.b;
            env = cand_env;
            dist_sum = cand_dsum;
            curE = newE;
            curS = newS;

            if(newS < bestS){
                bestS = newS;
                best = cur;
            }
        }
        T *= cool;
    }

    SAResult res;
    res.best = std::move(best);
    res.best_side = bestS;
    res.iters = iters;
    res.accepted = accepted_cnt;
    res.collided = collided_cnt;
    return res;
}

// ---- CSV helpers ----
static inline string strip_s(string s){
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    if(!s.empty() && s[0]=='s') s.erase(s.begin());
    return s;
}
static vector<string> split_csv_simple(const string& line){
    vector<string> out;
    string cur;
    for(char ch: line){
        if(ch==','){ out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    out.push_back(cur);
    return out;
}

struct Row { int item; ld x,y,deg; };

static string fmt(double v){
    ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << setprecision(18) << v;
    return oss.str();
}

// ---- dimer tiling helpers ----
static inline bool strict_no_overlaps(const vector<Tree>& ts){
    int n = (int)ts.size();
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            if(strict_overlap_candidate(ts[i], ts[j])) return false;
        }
    }
    return true;
}

static inline bool env_within(const Bounds& inner, const Bounds& outer, ld tol=EPS){
    return (inner.minx >= outer.minx - tol &&
            inner.miny >= outer.miny - tol &&
            inner.maxx <= outer.maxx + tol &&
            inner.maxy <= outer.maxy + tol);
}

static inline int max_steps_fit(ld total_len, ld obj_len, ld gap, ld tol){
    // 最大 n 使得 n*obj_len + (n-1)*gap <= total_len + tol
    if(obj_len <= 0) return 0;
    int n = 0;
    for(int k=1;k<=100000;k++){
        ld need = (ld)k * obj_len + (ld)(k-1) * gap;
        if(need <= total_len + tol) n = k;
        else break;
    }
    return n;
}

static bool tile_group_with_dimers(
    const vector<Row>& rows_sorted_by_item,
    vector<Row>& out_rows,          // same items, new x/y/deg
    int verbose
){
    const int n = (int)rows_sorted_by_item.size();
    if(n == 0) return false;

    // 原包络（按原解的 AABB 并集）
    vector<Tree> orig_ts; orig_ts.reserve(n);
    vector<Bounds> orig_bs; orig_bs.reserve(n);
    for(auto &r: rows_sorted_by_item){
        orig_ts.push_back(build_tree(r.x, r.y, r.deg));
        orig_bs.push_back(orig_ts.back().b);
    }
    Bounds orig_env = envelope_from_bounds(orig_bs);
    ld orig_side = side_len(orig_env);
    if(!(orig_side > 0)) return false;

    // 允许整个 group 选择一个全局角 phi（所有二聚体同角度平铺）
    const array<ld,5> GLOBAL_PHIS = {0.0L, 90.0L, 60.0L, 45.0L, 30.0L};

    const ld gap = 1e-12L;  // 额外网格余量
    const ld tol = 1e-12L;  // 判定容差

    // 需要的格子数：dimers + (odd?single:0)
    int n_dimers = n / 2;
    bool has_single = (n % 2) != 0;
    int needed_cells = n_dimers + (has_single ? 1 : 0);
    if(needed_cells <= 0) return false;

    struct Cand {
        ld side;
        ld phi;
        int total_cols;
        int rows_cnt;
        ld pack_w;
        ld pack_h;
        Bounds b_local;
        ld w;
        ld h;
    };
    vector<Cand> cands;
    cands.reserve((size_t)needed_cells * GLOBAL_PHIS.size());

    // 枚举全局角 + 列数，目标：pack_side 最小且 <= orig_side
    for(ld phi : GLOBAL_PHIS){
        Bounds b = Dimer::local_bounds(phi);
        ld w = b.maxx - b.minx;
        ld h = b.maxy - b.miny;
        if(w <= 0 || h <= 0) continue;
        for(int total_cols = 1; total_cols <= needed_cells; total_cols++){
            int rows_cnt = (needed_cells + total_cols - 1) / total_cols;
            ld pack_w = (ld)total_cols * w + (ld)(total_cols - 1) * gap;
            ld pack_h = (ld)rows_cnt * h + (ld)(rows_cnt - 1) * gap;
            ld pack_side = max(pack_w, pack_h);
            if(pack_side <= orig_side + tol){
                cands.push_back({pack_side, phi, total_cols, rows_cnt, pack_w, pack_h, b, w, h});
            }
        }
    }

    if(cands.empty()){
        if(verbose){
            cerr << "[dimer] no feasible grid under side constraint: orig_side=" << (double)orig_side
                 << " needed_cells=" << needed_cells << "\n";
        }
        return false;
    }
    sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){
        if(a.side != b.side) return a.side < b.side;
        if(a.total_cols != b.total_cols) return a.total_cols > b.total_cols; // side相同更偏好列多(更矮)
        return a.phi < b.phi;
    });

    for(const auto& cand : cands){
        ld phi = cand.phi;
        int total_cols = cand.total_cols;
        int rows_cnt = cand.rows_cnt;
        (void)rows_cnt;
        Bounds b = cand.b_local;
        ld w = cand.w;
        ld h = cand.h;

        // 预生成新 trees
        vector<Tree> new_ts; new_ts.reserve(n);
        out_rows.clear();
        out_rows.reserve(n);

        auto cell_origin = [&](int r, int c)->Vec{
            // 允许整体平移：把打包后的网格左下角对齐到 orig_env.min
            // 注意：side_len 对平移不敏感，此处仅为了让坐标范围与原解相近
            ld ox = orig_env.minx + (ld)c * (w + gap);
            ld oy = orig_env.miny + (ld)r * (h + gap);
            return {ox, oy};
        };

        auto place_dimer_in_cell = [&](int cell_idx, int itemA, int itemB)->bool{
            int r = cell_idx / total_cols;
            int c = cell_idx % total_cols;
            (void)c;
            Bounds lb = b;

            Vec org = cell_origin(r, c);
            // 让 dimer 的 local min 对齐到 cell origin
            ld ax = org.x - lb.minx;
            ld ay = org.y - lb.miny;
            auto ts = Dimer::build(ax, ay, phi);

            // 写入（按 item id 传入）
            out_rows.push_back({itemA, ts[0].cx, ts[0].cy, ts[0].ang_deg});
            out_rows.push_back({itemB, ts[1].cx, ts[1].cy, ts[1].ang_deg});
            new_ts.push_back(ts[0]);
            new_ts.push_back(ts[1]);
            return true;
        };

        auto place_single_in_cell = [&](int cell_idx, int item)->bool{
            int r = cell_idx / total_cols;
            int c = cell_idx % total_cols;
            (void)c;
            ld ang = Dimer::ANG_A0 + phi; // 单体用 A 的朝向
            Tree t0 = build_tree(0.0L, 0.0L, ang);
            Bounds lb = t0.b;
            Vec org = cell_origin(r, c);
            ld cx = org.x - lb.minx;
            ld cy = org.y - lb.miny;
            Tree t = build_tree(cx, cy, ang);
            out_rows.push_back({item, t.cx, t.cy, t.ang_deg});
            new_ts.push_back(t);
            return true;
        };

        // 放置 dimers（按 item 升序两两配对）
        for(int k=0;k<n_dimers;k++){
            int itemA = rows_sorted_by_item[2*k].item;
            int itemB = rows_sorted_by_item[2*k+1].item;
            if(!place_dimer_in_cell(k, itemA, itemB)){
                out_rows.clear();
                return false;
            }
        }
        if(has_single){
            int item = rows_sorted_by_item[2*n_dimers].item;
            if(!place_single_in_cell(n_dimers, item)){
                out_rows.clear();
                return false;
            }
        }

        // 验证：side_len 不变/不变大 + 严格无重叠
        vector<Bounds> new_bs; new_bs.reserve(new_ts.size());
        for(auto &t: new_ts) new_bs.push_back(t.b);
        Bounds new_env = envelope_from_bounds(new_bs);
        ld new_side = side_len(new_env);
        if(new_side > orig_side + tol){
            if(verbose) cerr << "[dimer] reject: side_len increased new=" << (double)new_side << " orig=" << (double)orig_side << "\n";
            continue;
        }
        if(!strict_no_overlaps(new_ts)){
            if(verbose) cerr << "[dimer] reject: overlaps\n";
            continue;
        }

        // 成功：按 item 排序输出 rows
        sort(out_rows.begin(), out_rows.end(), [](const Row& a, const Row& b){ return a.item < b.item; });
        return true;
    }

    return false;
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // base polygon (same as python)
    BASE_V = {{
        {0.0L,    0.8L},
        {0.125L,  0.5L},
        {0.0625L, 0.5L},
        {0.2L,    0.25L},
        {0.1L,    0.25L},
        {0.35L,   0.0L},
        {0.075L,  0.0L},
        {0.075L, -0.2L},
        {-0.075L,-0.2L},
        {-0.075L, 0.0L},
        {-0.35L,  0.0L},
        {-0.1L,   0.25L},
        {-0.2L,   0.25L},
        {-0.0625L,0.5L},
        {-0.125L, 0.5L}
    }};
    compute_base_centroid();

    // ---- 模式选择 ----
    // 默认：原模拟退火模式
    // 新增：--dimer 输入 输出 [verbose=0] [n_min=5] [n_max=200]
    bool dimer_mode = (argc >= 2 && string(argv[1]) == "--dimer");

    string in_path, out_path;
    int verbose = 0;
    int n_min = 5, n_max = 200;

    // SA 参数
    int max_iter = 1000000;
    double total_time_sec = 0.0;
    double per_group_sec = 0.0;
    uint64_t seed = 123456789ULL;
    const ld T_START = 1.2L;
    const ld T_END   = 0.001L;

    if(dimer_mode){
        in_path  = (argc>=3 ? argv[2] : "./submission.csv");
        out_path = (argc>=4 ? argv[3] : "./submission_dimer.csv");
        verbose  = (argc>=5 ? stoi(argv[4]) : 0);
        n_min    = (argc>=6 ? stoi(argv[5]) : 5);
        n_max    = (argc>=7 ? stoi(argv[6]) : 200);
    }else{
        // 命令行参数：输入输出路径、最大迭代、全局/单组时间、种子、日志、n 范围
        in_path  = (argc>=2 ? argv[1] : "./submission.csv");
        out_path = (argc>=3 ? argv[2] : "./summission_sa.csv");
        max_iter = (argc>=4 ? stoi(argv[3]) : 1000000);          // 每组基础迭代
        total_time_sec = (argc>=5 ? stod(argv[4]) : 0.0);        // 全局时间限制（0=无限）
        per_group_sec  = (argc>=6 ? stod(argv[5]) : 0.0);        // 单组时间限制（0=无限）
        seed = (argc>=7 ? (uint64_t)stoull(argv[6]) : 123456789ULL); // 随机种子
        verbose = (argc>=8 ? stoi(argv[7]) : 0);                 // 日志开关
        n_min   = (argc>=9 ? stoi(argv[8]) : 5);                 // 组大小下限
        n_max   = (argc>=10 ? stoi(argv[9]) : 200);              // 组大小上限
    }

    // read CSV
    ifstream ifs(in_path);
    if(!ifs){
        cerr << "Failed to open: " << in_path << "\n";
        return 1;
    }
    string header;
    getline(ifs, header);

    unordered_map<int, vector<Row>> groups; // gid(int) -> rows
    string line;
    while(getline(ifs, line)){
        if(line.empty()) continue;
        auto cols = split_csv_simple(line);
        if(cols.size() < 4) continue;

        string id = cols[0];
        auto pos = id.find('_');
        if(pos==string::npos) continue;
        int gid = stoi(id.substr(0,pos));
        int item = stoi(id.substr(pos+1));

        ld x = (ld)stold(strip_s(cols[1]));
        ld y = (ld)stold(strip_s(cols[2]));
        ld deg = (ld)stold(strip_s(cols[3]));
        groups[gid].push_back({item, x, y, deg});
    }

    if(dimer_mode){
        unordered_map<int, vector<Row>> out_groups = groups; // 默认保留原解，成功则覆盖

        vector<int> gids;
        gids.reserve(groups.size());
        for(auto &kv: groups) gids.push_back(kv.first);
        sort(gids.begin(), gids.end());

        int changed = 0, tried = 0;
        for(int gid: gids){
            auto rows = groups[gid];
            sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.item < b.item; });
            int n = (int)rows.size();
            if(n < n_min || n > n_max) continue;

            tried++;
            vector<Row> new_rows;
            bool ok = tile_group_with_dimers(rows, new_rows, verbose);
            if(ok){
                out_groups[gid] = std::move(new_rows);
                changed++;
            }
            if(verbose){
                cerr << "[dimer] gid=" << gid << " n=" << n << " ok=" << ok << "\n";
            }
        }

        // write CSV（保持 gid 升序，组内 item 升序）
        ofstream ofs(out_path);
        if(!ofs){
            cerr << "Failed to write: " << out_path << "\n";
            return 1;
        }
        ofs << "id,x,y,deg\n";
        for(int gid: gids){
            auto rows = out_groups[gid];
            sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.item < b.item; });
            for(auto &r: rows){
                ofs << setw(3) << setfill('0') << gid << "_" << r.item << ","
                    << "s" << fmt((double)r.x) << ","
                    << "s" << fmt((double)r.y) << ","
                    << "s" << fmt((double)r.deg) << "\n";
            }
        }
        if(verbose){
            cerr << "[dimer] tried=" << tried << " changed=" << changed
                 << " wrote=" << out_path << "\n";
        }
        return 0;
    }

    int hw_threads = (int)std::thread::hardware_concurrency(); // 使用硬件线程数
    if(hw_threads <= 0) hw_threads = 1;
#ifdef _OPENMP
    omp_set_num_threads(hw_threads);
    hw_threads = omp_get_max_threads();
#endif

    auto t_global0 = chrono::steady_clock::now();

    // prepare output rows
    unordered_map<int, vector<Row>> out_groups = groups; // 默认保留原解，改进则替换

    vector<int> gid_list; // 记录分组 id，排序后迭代
    gid_list.reserve(groups.size());
    for (auto &kv : groups) gid_list.push_back(kv.first);
    sort(gid_list.begin(), gid_list.end());

    struct GroupResult { int gid; vector<Row> rows; bool done=false; };
    vector<GroupResult> results(gid_list.size()); // 存并行结果
    atomic<bool> stop_flag(false);                // 全局超时标记

#pragma omp parallel for schedule(dynamic) // 按组动态分配并行
    for(int idx=0; idx<(int)gid_list.size(); idx++){
        if(stop_flag.load()) continue;
        int gid = gid_list[idx];

        auto rows = groups[gid];
        sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.item < b.item; });
        int n = (int)rows.size();

        if(n < n_min || n > n_max) continue;

        if(total_time_sec > 0){ // 全局时间限制
            double eg = chrono::duration<double>(chrono::steady_clock::now()-t_global0).count();
            if(eg > total_time_sec){
                stop_flag.store(true);
                continue;
            }
        }

        vector<Tree> init; // 构造当前组初始解
        init.reserve(n);
        for(auto &r: rows){
            init.push_back(build_tree(r.x, r.y, r.deg));
        }

        vector<Bounds> bs; bs.reserve(n);
        for(auto &t: init) bs.push_back(t.b);
        ld origS = side_len(envelope_from_bounds(bs)); // 原始侧长（未直接使用）
        (void)origS;

        uint64_t salt = (uint64_t)gid * 0x9E3779B97F4A7C15ULL; // 不同组/线程派生种子
#ifdef _OPENMP
        salt += (uint64_t)omp_get_thread_num();
#endif
        SplitMix64 local_rng(seed + salt);

        auto res = run_sa_group(init, max_iter, T_START, T_END, local_rng, per_group_sec, verbose);

        vector<Row> new_rows;
        new_rows.reserve(n);
        for(int i=0;i<n;i++){
            new_rows.push_back({rows[i].item, res.best[i].cx, res.best[i].cy, res.best[i].ang_deg});
        }
        results[idx] = {gid, std::move(new_rows), true};
    }

    for(size_t i=0;i<results.size();i++){
        if(results[i].done){
            out_groups[results[i].gid] = std::move(results[i].rows);
        }
    }

    // write CSV（保持 gid 升序，组内 item 升序）
    ofstream ofs(out_path);
    if(!ofs){
        cerr << "Failed to write: " << out_path << "\n";
        return 1;
    }
    ofs << "id,x,y,deg\n";

    vector<int> gids;
    gids.reserve(out_groups.size());
    for(auto &kv: out_groups) gids.push_back(kv.first);
    sort(gids.begin(), gids.end());

    for(int gid: gids){
        auto rows = out_groups[gid];
        sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.item < b.item; });
        for(auto &r: rows){
            ofs << setw(3) << setfill('0') << gid << "_" << r.item << ","
                << "s" << fmt((double)r.x) << ","
                << "s" << fmt((double)r.y) << ","
                << "s" << fmt((double)r.deg) << "\n";
        }
    }

    return 0;
}