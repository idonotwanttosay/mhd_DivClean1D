#include "solver.hpp"
#include "physics.hpp"
#include "io.hpp"

#include <filesystem>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

static std::string prepare_output_dir(){
    namespace fs = std::filesystem;
    fs::path base("Result");
    if(fs::exists(base) && !fs::is_empty(base)){
        auto ts = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        fs::rename(base, "Result_"+std::to_string(ts));
    }
    fs::create_directory(base);
    return "Result";
}

int main(){
    const int nx = 256, ny = 256;
    const double Lx = 2.0, Ly = 2.0; // domain [-1,1]x[-1,1]
    const double dx = Lx/(nx-1), dy = Ly/(ny-1);
    const double cfl = 0.3;
    const double t_end = 0.25;
    const int output_every = 20;

    std::string out_dir = prepare_output_dir();

    AMRGrid amr(nx, ny, Lx, Ly, 1); // single level grid
    FlowField flow(nx, ny, dx, dy, -1.0, -1.0);
    initialize_2d_riemann(flow);
    std::vector<FlowField> flows = {flow};

    auto t0 = std::chrono::high_resolution_clock::now();
    double t = 0.0;
    int step = 0;
    while (t < t_end) {
        double dt = compute_cfl_timestep(flows[0], cfl);
        if (t + dt > t_end) dt = t_end - t;

        solve_MHD(amr, flows, dt, 0.0, 0, 0.0);

        t += dt;
        step++;

        if (step % output_every == 0) {
            auto [max_divB, L1_divB] = compute_divergence_errors(flows[0]);
            std::cout << "step " << std::setw(4) << step
                      << " t=" << t
                      << " dt=" << dt
                      << " max_divB=" << max_divB
                      << " L1_divB=" << L1_divB << "\n";
            save_flow_MHD(flows[0], out_dir, step);
        }
    }
    auto t1=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "Total time " << elapsed.count() << " s\n";
    return 0;
}
