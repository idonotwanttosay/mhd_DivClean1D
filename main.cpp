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
    const int nx=256, ny=256; // higher resolution for peak Bx test
    const double Lx=1.0, Ly=1.0;
    const double dx=Lx/(nx-1), dy=Ly/(ny-1);
    const int max_steps=1000;
    const int output_every=20;

    std::string out_dir = prepare_output_dir();

    AMRGrid amr(nx, ny, Lx, Ly, 1); // single level grid
    FlowField flow(nx,ny,dx,dy);
    initialize_peak_bx(flow);      // peak Bx initial condition
    std::vector<FlowField> flows = {flow};

    auto t0=std::chrono::high_resolution_clock::now();
    for(int step=0; step<=max_steps; ++step){
        double dt = compute_cfl_timestep(flows[0]);

        solve_MHD(amr, flows, dt, 0.0, 0, 0.0); // full MHD update

        if(step % output_every == 0){
            auto [max_divB, L1_divB] = compute_divergence_errors(flows[0]);
            std::cout << "step " << std::setw(4) << step
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
