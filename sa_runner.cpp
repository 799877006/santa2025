// sa_runner.cpp
#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

using ld = long double;

struct Vec { ld x, y; };
static inline Vec operator+(const Vec& a, const Vec& b){ return {a.x+b.x, a.y+b.y}; }
static inline Vec operator-(const Vec& a, const Vec& b){ return {a.x-b.x, a.y-b.y}; }
static inline ld cross(const Vec& a, const Vec& b){ return a.x*b.y - a.y*b.x; }

struct Bounds { ld minx, miny, maxx, maxy; };

static constexpr int NV = 15;
static constexpr ld EPS = 1e-12L;

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

static inline int sgn(ld v){
    if (v > EPS) return 1;
    if (v < -EPS) return -1;
    return 0;
}

static inline bool on_segment(const Vec& a, const Vec& b, const Vec& p){
    if (fabsl(cross(b-a, p-a)) > EPS) return false;
    return (min(a.x,b.x) - EPS <= p.x && p.x <= max(a.x,b.x) + EPS &&
            min(a.y,b.y) - EPS <= p.y && p.y <= max(a.y,b.y) + EPS);
}

// strict crossing only
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

// 1 strict inside, 0 boundary, -1 outside
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

static inline bool aabb_disjoint(const Bounds& A, const Bounds& B){
    return (A.maxx < B.minx || A.minx > B.maxx || A.maxy < B.miny || A.miny > B.maxy);
}

static array<Vec,NV> BASE_V;
static Vec BASE_CENTROID;

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

struct Tree {
    ld cx, cy, ang_deg;
    array<Vec,NV> v;
    Bounds b;
    Vec centroid;
};

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
static inline ld side_len(const Bounds& e){
    return max(e.maxx - e.minx, e.maxy - e.miny);
}

struct SAResult {
    vector<Tree> best;
    ld best_side;
    long long iters=0, accepted=0, collided=0;
};

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
    bool small = (n <= 50);

    int iters = small ? max_iterations * 3 : max_iterations;
    ld T0 = small ? t_start * 2.0L : t_start;
    ld gravity_w = small ? 1e-4L : 1e-6L;

    vector<Tree> cur = initial;
    vector<Bounds> cur_bounds(n);
    for(int i=0;i<n;i++) cur_bounds[i] = cur[i].b;

    Bounds env = envelope_from_bounds(cur_bounds);

    ld dist_sum = 0;
    for(int i=0;i<n;i++) dist_sum += cur[i].cx*cur[i].cx + cur[i].cy*cur[i].cy;

    auto energy = [&](const Bounds& e, ld dsum)->pair<ld, ld>{
        ld s = side_len(e);
        ld norm_d = dsum / (ld)max(1,n);
        ld E = s + gravity_w * norm_d;
        return {E, s};
    };

    auto [curE, curS] = energy(env, dist_sum);

    vector<Tree> best = cur;
    ld bestS = curS;

    ld cool = powl(t_end / T0, 1.0L / (ld)iters);
    ld T = T0;

    auto t0 = chrono::steady_clock::now();

    long long accepted_cnt=0, collided_cnt=0;
    for(int it=0; it<iters; it++){
        if(per_group_time_sec > 0){
            double el = chrono::duration<double>(chrono::steady_clock::now()-t0).count();
            if(el > per_group_time_sec) break;
        }
        if(verbose && (it % 200000 == 0)){
            cout << "[sa] it=" << it << "/" << iters
                 << " T=" << (double)T
                 << " curS=" << (double)curS
                 << " bestS=" << (double)bestS
                 << " acc=" << accepted_cnt
                 << " col=" << collided_cnt
                 << "\n" << flush;
        }

        ld progress = (ld)it / (ld)iters;

        ld move_scale, rot_scale;
        if(small){
            move_scale = max((ld)0.005L, (ld)3.0L * (1.0L - progress));
            rot_scale  = max((ld)0.001L, (ld)5.0L * (1.0L - progress));
        }else{
            move_scale = max((ld)0.001L, (ld)1.0L * (T / T0));
            rot_scale  = max((ld)0.002L, (ld)5.0L * (T / T0));
        }

        int idx = rng.randint(0, n-1);
        const Tree& orig = cur[idx];
        const Bounds& ob = cur_bounds[idx];

        ld dx = (rng.next01() - 0.5L) * 0.1L * move_scale;
        ld dy = (rng.next01() - 0.5L) * 0.1L * move_scale;
        ld dang = (rng.next01() - 0.5L) * rot_scale;

        ld ncx = orig.cx + dx;
        ld ncy = orig.cy + dy;
        ld nang = orig.ang_deg + dang;

        Tree cand = build_tree(ncx, ncy, nang);

        bool collision = false;
        for(int k=0;k<n;k++){
            if(k==idx) continue;
            if(aabb_disjoint(cand.b, cur_bounds[k])) continue;
            if(strict_overlap_candidate(cand, cur[k])){
                collision = true;
                break;
            }
        }
        if(collision){
            collided_cnt++;
            T *= cool;
            continue;
        }

        ld old_d = orig.cx*orig.cx + orig.cy*orig.cy;
        ld new_d = ncx*ncx + ncy*ncy;
        ld cand_dsum = dist_sum - old_d + new_d;

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

    string in_path  = (argc>=2 ? argv[1] : "./submission.csv");
    string out_path = (argc>=3 ? argv[2] : "./summission_sa.csv");

    int max_iter = (argc>=4 ? stoi(argv[3]) : 1000000);
    double total_time_sec = (argc>=5 ? stod(argv[4]) : 0.0);     // 0=無制限
    double per_group_sec  = (argc>=6 ? stod(argv[5]) : 0.0);     // 0=無制限
    uint64_t seed = (argc>=7 ? (uint64_t)stoull(argv[6]) : 123456789ULL);
    int verbose = (argc>=8 ? stoi(argv[7]) : 0);
    int n_min   = (argc>=9 ? stoi(argv[8]) : 5);
    int n_max   = (argc>=10 ? stoi(argv[9]) : 200);

    const ld T_START = 1.0L;
    const ld T_END   = 0.001L;

    // read
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

    int hw_threads = (int)std::thread::hardware_concurrency();
    if(hw_threads <= 0) hw_threads = 1;
#ifdef _OPENMP
    omp_set_num_threads(hw_threads);
    hw_threads = omp_get_max_threads();
#endif

    auto t_global0 = chrono::steady_clock::now();

    // prepare output rows
    unordered_map<int, vector<Row>> out_groups = groups;

    vector<int> gid_list;
    gid_list.reserve(groups.size());
    for (auto &kv : groups) gid_list.push_back(kv.first);
    sort(gid_list.begin(), gid_list.end());

    struct GroupResult { int gid; vector<Row> rows; bool done=false; };
    vector<GroupResult> results(gid_list.size());
    atomic<bool> stop_flag(false);

#pragma omp parallel for schedule(dynamic)
    for(int idx=0; idx<(int)gid_list.size(); idx++){
        if(stop_flag.load()) continue;
        int gid = gid_list[idx];

        auto rows = groups[gid];
        sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.item < b.item; });
        int n = (int)rows.size();

        if(n < n_min || n > n_max) continue;

        if(total_time_sec > 0){
            double eg = chrono::duration<double>(chrono::steady_clock::now()-t_global0).count();
            if(eg > total_time_sec){
                stop_flag.store(true);
                continue;
            }
        }

        vector<Tree> init;
        init.reserve(n);
        for(auto &r: rows){
            init.push_back(build_tree(r.x, r.y, r.deg));
        }

        vector<Bounds> bs; bs.reserve(n);
        for(auto &t: init) bs.push_back(t.b);
        ld origS = side_len(envelope_from_bounds(bs));
        (void)origS;

        uint64_t salt = (uint64_t)gid * 0x9E3779B97F4A7C15ULL;
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

    // write CSV
    // keep all groups, stable order: gid asc, item asc
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