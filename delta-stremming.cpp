#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>

using namespace std;

const int INF = 1000000000;

struct Edge {
    int to, w;
};

struct Graph {
    int n;
    vector<vector<Edge>> adj;
    Graph(int n = 0) : n(n), adj(n) {}
};

inline int owner(int v, int size, int n) {
    return (int)((1LL * v * size) / n);
}

double comm_time = 0.0;

struct EdgeTriplet {
    int u, v, w;
};

string getGraphFilename(int n, double density, int rank, int size) {
    string filename =
        "graph_n" + to_string(n) +
        "_d" + to_string(static_cast<int>(density * 100)) +
        "_p" + to_string(size) +
        "_r" + to_string(rank) + ".bin";
    return filename;
}

void saveLocalGraph(const Graph& g, int rank, int size, int n, double density) {
    string filename = getGraphFilename(n, density, rank, size);
    ofstream file(filename, ios::binary);

    if (!file) {
        cerr << "Rank " << rank << ": Error opening file " << filename << " for writing\n";
        return;
    }

    file.write(reinterpret_cast<const char*>(&n), sizeof(int));

    int first = (rank * n) / size;
    int last = ((rank + 1) * n) / size;
    file.write(reinterpret_cast<const char*>(&first), sizeof(int));
    file.write(reinterpret_cast<const char*>(&last), sizeof(int));

    for (int u = first; u < last; u++) {
        int edge_count = (int)g.adj[u].size();
        file.write(reinterpret_cast<const char*>(&edge_count), sizeof(int));
        for (size_t i = 0; i < g.adj[u].size(); ++i) {
            int to = g.adj[u][i].to;
            int w = g.adj[u][i].w;
            file.write(reinterpret_cast<const char*>(&to), sizeof(int));
            file.write(reinterpret_cast<const char*>(&w), sizeof(int));
        }
    }

    file.close();
    if (rank == 0) cout << "Saved local graph to " << filename << "\n";
}

bool loadLocalGraph(Graph& g, int rank, int size, int n, double density) {
    string filename = getGraphFilename(n, density, rank, size);
    ifstream file(filename, ios::binary);

    if (!file) {
        if (rank == 0) cout << "Graph file not found: " << filename << "\n";
        return false;
    }

    int file_n = 0;
    file.read(reinterpret_cast<char*>(&file_n), sizeof(int));
    if (file_n != n) {
        cerr << "Rank " << rank << ": Graph size mismatch. Expected " << n
            << ", got " << file_n << "\n";
        return false;
    }

    int first = 0, last = 0;
    file.read(reinterpret_cast<char*>(&first), sizeof(int));
    file.read(reinterpret_cast<char*>(&last), sizeof(int));

    int expected_first = (rank * n) / size;
    int expected_last = ((rank + 1) * n) / size;
    if (first != expected_first || last != expected_last) {
        cerr << "Rank " << rank << ": Vertex range mismatch\n";
        return false;
    }

    g = Graph(n);

    for (int u = first; u < last; u++) {
        int edge_count = 0;
        file.read(reinterpret_cast<char*>(&edge_count), sizeof(int));
        g.adj[u].reserve(edge_count);
        for (int i = 0; i < edge_count; i++) {
            int to = 0, w = 0;
            file.read(reinterpret_cast<char*>(&to), sizeof(int));
            file.read(reinterpret_cast<char*>(&w), sizeof(int));
            g.adj[u].push_back({ to, w });
        }
    }

    file.close();
    if (rank == 0) cout << "Loaded local graph from " << filename << "\n";
    return true;
}

bool checkAllGraphFilesExist(int n, double density, int size) {
    for (int rank = 0; rank < size; rank++) {
        string filename = getGraphFilename(n, density, rank, size);
        ifstream file(filename, ios::binary);
        if (!file) return false;
        file.close();
    }
    return true;
}

// =============================================================
// Генерация/загрузка распределённого графа
// ШАГ 1: генерация распределённая:
// =============================================================
Graph generateOrLoadDistributedGraph(int n, double density, MPI_Comm comm,
    bool& loaded_from_file, double& gen_time) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Graph g(n);
    loaded_from_file = false;

    MPI_Barrier(comm);

    bool all_files_exist = checkAllGraphFilesExist(n, density, size);
    MPI_Bcast(&all_files_exist, 1, MPI_C_BOOL, 0, comm);

    if (all_files_exist) {
        if (rank == 0) cout << "Loading graph from files\n";

        double load_start = MPI_Wtime();
        bool load_success = loadLocalGraph(g, rank, size, n, density);

        bool all_loaded = false;
        MPI_Allreduce(&load_success, &all_loaded, 1, MPI_C_BOOL, MPI_LAND, comm);

        if (all_loaded) {
            loaded_from_file = true;
            gen_time = MPI_Wtime() - load_start;
            if (rank == 0) cout << "graph loaded successfully from files\n";
            return g;
        }
        else {
            if (rank == 0) cout << "failed to load graph from files, generating new...\n";
        }
    }

    if (rank == 0) cout << "generating new graph\n";

    double gen_start = MPI_Wtime();

    int first = (rank * n) / size;
    int last = ((rank + 1) * n) / size;

    random_device rd;
    mt19937 gen(static_cast<unsigned>(rd()) + rank);
    uniform_int_distribution<> weight_dist(1, 100);
    uniform_real_distribution<> prob(0.0, 1.0);

    vector<vector<EdgeTriplet>> send_buf(size); // только обратные дуги (v->u) владельцу v

    for (int u = first; u < last; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (prob(gen) < density) {
                int w = weight_dist(gen);

                // (u -> v) локально (owner(u) == rank)
                g.adj[u].push_back({ v, w });

                // (v -> u) владельцу v
                int ov = owner(v, size, n);
                if (ov == rank) {
                    g.adj[v].push_back({ u, w });
                }
                else {
                    send_buf[ov].push_back(EdgeTriplet{ v, u, w });
                }
            }
        }
    }

    // обмен обратными дугами
    vector<int> send_counts(size, 0), recv_counts(size, 0);
    for (int i = 0; i < size; ++i) send_counts[i] = (int)send_buf[i].size();

    double t = MPI_Wtime();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
        recv_counts.data(), 1, MPI_INT, comm);
    comm_time += MPI_Wtime() - t;

    vector<int> send_counts_int(size, 0), recv_counts_int(size, 0);
    for (int i = 0; i < size; ++i) {
        send_counts_int[i] = send_counts[i] * 3;
        recv_counts_int[i] = recv_counts[i] * 3;
    }

    vector<int> send_disp_int(size, 0), recv_disp_int(size, 0);
    int send_total_int = 0, recv_total_int = 0;
    for (int i = 0; i < size; ++i) {
        send_disp_int[i] = send_total_int;
        send_total_int += send_counts_int[i];

        recv_disp_int[i] = recv_total_int;
        recv_total_int += recv_counts_int[i];
    }

    vector<int> send_flat(send_total_int);
    vector<int> recv_flat(recv_total_int);

    int pos = 0;
    for (int proc = 0; proc < size; ++proc) {
        for (size_t k = 0; k < send_buf[proc].size(); ++k) {
            send_flat[pos++] = send_buf[proc][k].u;
            send_flat[pos++] = send_buf[proc][k].v;
            send_flat[pos++] = send_buf[proc][k].w;
        }
    }

    t = MPI_Wtime();
    MPI_Alltoallv(send_flat.data(), send_counts_int.data(), send_disp_int.data(), MPI_INT,
        recv_flat.data(), recv_counts_int.data(), recv_disp_int.data(), MPI_INT,
        comm);
    comm_time += MPI_Wtime() - t;

    // распаковка
    for (int i = 0; i < recv_total_int; i += 3) {
        int u = recv_flat[i];
        int v = recv_flat[i + 1];
        int w = recv_flat[i + 2];
        if (u >= first && u < last) {
            g.adj[u].push_back({ v, w });
        }
    }

    gen_time = MPI_Wtime() - gen_start;

    saveLocalGraph(g, rank, size, n, density);

    if (rank == 0) cout << "New graph generated and saved\n";
    return g;
}


static void exchange_relax_requests_alltoallv(
    const vector<vector<pair<int, int>>>& send_req_in,
    vector<pair<int, int>>& recv_req,
    MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    vector<int> send_counts(size, 0), recv_counts(size, 0);
    for (int p = 0; p < size; ++p) send_counts[p] = (int)send_req_in[p].size();
    double t = MPI_Wtime();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
        recv_counts.data(), 1, MPI_INT, comm);
    comm_time += MPI_Wtime() - t;
    vector<int> send_counts_int(size, 0), recv_counts_int(size, 0);
    for (int p = 0; p < size; ++p) {
        send_counts_int[p] = send_counts[p] * 2;
        recv_counts_int[p] = recv_counts[p] * 2;}
    vector<int> sdisp(size, 0), rdisp(size, 0);
    int stotal = 0, rtotal = 0;
    for (int p = 0; p < size; ++p) {
        sdisp[p] = stotal; stotal += send_counts_int[p];
        rdisp[p] = rtotal; rtotal += recv_counts_int[p];}
    vector<int> send_flat(stotal);
    vector<int> recv_flat(rtotal);
    int pos = 0;
    for (int p = 0; p < size; ++p) {
        for (size_t k = 0; k < send_req_in[p].size(); ++k) {
            send_flat[pos++] = send_req_in[p][k].first;   // v
            send_flat[pos++] = send_req_in[p][k].second;  // cand
        }}
    t = MPI_Wtime();
    MPI_Alltoallv(send_flat.data(), send_counts_int.data(), sdisp.data(), MPI_INT,
        recv_flat.data(), recv_counts_int.data(), rdisp.data(), MPI_INT,
        comm);
    comm_time += MPI_Wtime() - t;

    recv_req.clear();
    recv_req.reserve(rtotal / 2);
    for (int i = 0; i < rtotal; i += 2) {
        recv_req.push_back(make_pair(recv_flat[i], recv_flat[i + 1]));
    }
}

static void push_min_req(vector<unordered_map<int, int>>& best,
    int proc, int v, int cand) {
    unordered_map<int, int>::iterator it = best[proc].find(v);
    if (it == best[proc].end() || cand < it->second) {
        best[proc][v] = cand;
    }
}

static void maps_to_sendreq(const vector<unordered_map<int, int>>& best,
    vector<vector<pair<int, int>>>& send_req) {
    int size = (int)best.size();
    send_req.assign(size, vector<pair<int, int>>());
    for (int p = 0; p < size; ++p) {
        send_req[p].reserve(best[p].size());
        for (unordered_map<int, int>::const_iterator it = best[p].begin(); it != best[p].end(); ++it) {
            send_req[p].push_back(make_pair(it->first, it->second));
        }
    }
}

void delta_stepping_mpi_owner_compute(Graph& g, int source, int delta,
    vector<int>& dist_full_on_root,
    MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int n = g.n;

    int first = (rank * n) / size;
    int last = ((rank + 1) * n) / size;
    int local_n = last - first;

    auto is_owned = [&](int v) { return v >= first && v < last; };
    auto lid = [&](int v) { return v - first; };

    vector<int> dist_local(local_n, INF);
    if (is_owned(source)) dist_local[lid(source)] = 0;

    vector<vector<int>> buckets(1);
    auto ensure_bucket = [&](int b) {
        if (b >= (int)buckets.size()) buckets.resize(b + 1);
    };
    auto bucket_insert_owned = [&](int v, int d) {
        int b = d / delta;
        ensure_bucket(b);
        buckets[b].push_back(v);
    };

    if (is_owned(source)) {
        ensure_bucket(0);
        buckets[0].push_back(source);
    }

    while (true) {
        int local_min = INF;
        for (int i = 0; i < (int)buckets.size(); ++i) {
            if (!buckets[i].empty()) { local_min = i; break; }
        }

        int i_min;
        double t = MPI_Wtime();
        MPI_Allreduce(&local_min, &i_min, 1, MPI_INT, MPI_MIN, comm);
        comm_time += MPI_Wtime() - t;

        if (i_min == INF) break;

        ensure_bucket(i_min);

        vector<int> S_local = std::move(buckets[i_min]);
        buckets[i_min].clear();

        vector<int> R_local;
        R_local.reserve(S_local.size());

        // ---------- LIGHT ----------
        while (true) {
            for (size_t k = 0; k < S_local.size(); ++k) {
                int u = S_local[k];
                if (is_owned(u)) R_local.push_back(u);
            }
            vector<unordered_map<int, int>> best(size);
            for (int p = 0; p < size; ++p) best[p].reserve(256);

            for (size_t k = 0; k < S_local.size(); ++k) {
                int u = S_local[k];
                if (!is_owned(u)) continue;
                int du = dist_local[lid(u)];
                if (du >= INF) continue;
                for (size_t j = 0; j < g.adj[u].size(); ++j) {
                    const Edge& e = g.adj[u][j];
                    if (e.w <= delta) {
                        int v = e.to;
                        int cand = du + e.w;
                        int ov = owner(v, size, n);
                        push_min_req(best, ov, v, cand);}}}
            vector<vector<pair<int, int>>> send_req;
            maps_to_sendreq(best, send_req);

            vector<pair<int, int>> recv_req;
            exchange_relax_requests_alltoallv(send_req, recv_req, comm);

            vector<int> nextS_local;
            nextS_local.reserve(64);

            int changed_local = 0;

            for (size_t k = 0; k < recv_req.size(); ++k) {
                int v = recv_req[k].first;
                int cand = recv_req[k].second;

                if (!is_owned(v)) continue;
                int& dv = dist_local[lid(v)];
                if (cand < dv) {
                    dv = cand;
                    int b = dv / delta;
                    ensure_bucket(b);
                    buckets[b].push_back(v);
                    if (b == i_min) {
                        nextS_local.push_back(v);
                        changed_local = 1;
                    }
                }
            }

            int changed_global = 0;
            t = MPI_Wtime();
            MPI_Allreduce(&changed_local, &changed_global, 1, MPI_INT, MPI_LOR, comm);
            comm_time += MPI_Wtime() - t;

            S_local = std::move(nextS_local);

            if (!changed_global) break;
        }

        // ---------- HEAVY ----------
        {
            vector<unordered_map<int, int>> best(size);
            for (int p = 0; p < size; ++p) best[p].reserve(256);
            for (size_t k = 0; k < R_local.size(); ++k) {
                int u = R_local[k];
                if (!is_owned(u)) continue;

                int du = dist_local[lid(u)];
                if (du >= INF) continue;

                for (size_t j = 0; j < g.adj[u].size(); ++j) {
                    const Edge& e = g.adj[u][j];
                    if (e.w > delta) {
                        int v = e.to;
                        int cand = du + e.w;
                        int ov = owner(v, size, n);
                        push_min_req(best, ov, v, cand);}}}
            vector<vector<pair<int, int>>> send_req;
            maps_to_sendreq(best, send_req);
            vector<pair<int, int>> recv_req;
            exchange_relax_requests_alltoallv(send_req, recv_req, comm);
            for (size_t k = 0; k < recv_req.size(); ++k) {
                int v = recv_req[k].first;
                int cand = recv_req[k].second;
                if (!is_owned(v)) continue;
                int& dv = dist_local[lid(v)];
                if (cand < dv) {
                    dv = cand;
                    bucket_insert_owned(v, dv);
                }}}}

    // ---------- Gatherv dist на root ----------
    vector<int> recvcounts, displs;
    if (rank == 0) {
        dist_full_on_root.assign(n, INF);
        recvcounts.resize(size);
        displs.resize(size);
        for (int p = 0; p < size; ++p) {
            int f = (p * n) / size;
            int l = ((p + 1) * n) / size;
            recvcounts[p] = l - f;
            displs[p] = f;
        }
    }

    MPI_Gatherv(dist_local.data(), local_n, MPI_INT,
        (rank == 0 ? dist_full_on_root.data() : NULL),
        (rank == 0 ? recvcounts.data() : NULL),
        (rank == 0 ? displs.data() : NULL),
        MPI_INT, 0, comm);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    double total_start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    vector<int> graph_sizes = { 100, 1000, 5000, 10000 };
    double density = 0.3;
    int delta = 10;
    int source = 0;

    if (argc > 1) {
        int custom_n = atoi(argv[1]);
        graph_sizes = { custom_n };
    }
    if (argc > 2) density = atof(argv[2]);
    if (argc > 3) delta = atoi(argv[3]);

    for (size_t idx = 0; idx < graph_sizes.size(); ++idx) {
        int n = graph_sizes[idx];

        if (rank == 0) {
            cout << "\n=========================================\n";
            cout << "Testing with n = " << n
                << ", density = " << density
                << ", processes = " << size << "\n";
            cout << "=========================================\n";
        }

        MPI_Barrier(comm);

        bool loaded_from_file = false;
        double gen_time = 0.0;

        Graph g = generateOrLoadDistributedGraph(n, density, comm, loaded_from_file, gen_time);

        vector<int> dist_on_root;

        MPI_Barrier(comm);
        double algo_start = MPI_Wtime();

        delta_stepping_mpi_owner_compute(g, source, delta, dist_on_root, comm);

        double algo_time = MPI_Wtime() - algo_start;
        double total_time = MPI_Wtime() - total_start;

        double comm_sum = 0.0, comm_max = 0.0;
        MPI_Reduce(&comm_time, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (rank == 0) {
            cout << "Results for n = " << n << ":\n";
            cout << "  Graph source: " << (loaded_from_file ? "file" : "generated") << "\n";
            cout << "  Graph generation/load time: " << gen_time << " sec\n";
            cout << "  Delta-step (owner-compute) time: " << algo_time << " sec\n";
            cout << "  Communication time (sum over ranks): " << comm_sum << " sec\n";
            cout << "  Communication time (max rank): " << comm_max << " sec\n";
            cout << "  Total time: " << total_time << " sec\n";

            int print_count = min(n, 10);
            cout << "  First " << print_count << " distances: ";
            for (int i = 0; i < print_count; i++) {
                cout << dist_on_root[i] << " ";
            }
            cout << "\n";
        }

        comm_time = 0.0;
        MPI_Barrier(comm);
    }

    MPI_Finalize();
    return 0;
}
