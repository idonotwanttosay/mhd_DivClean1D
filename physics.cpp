#include "physics.hpp"
#include <random>
#include <cmath>
#include <cstdlib>

void initialize_MHD_disk(FlowField& flow,int seed)
{
    // allow override via env SEED
    const char* env = std::getenv("SEED");
    if(env){ seed = std::atoi(env); }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noise(-0.01,0.01);

    const double cs=0.1;
    const double gamma=1.4;

#pragma omp parallel for collapse(2)
    for(int i=0;i<flow.rho.nx;++i)
        for(int j=0;j<flow.rho.ny;++j){
            double x = flow.rho.x0 + i*flow.rho.dx - 0.5;
            double y = flow.rho.y0 + j*flow.rho.dy - 0.5;
            double r = std::sqrt(x*x+y*y)+1e-6;
            flow.rho.data[i][j]=1.0/(r*r+0.1);

            double vth=std::sqrt(1.0/std::max(r,0.01));
            flow.u.data[i][j]=-y/r*vth + noise(rng);
            flow.v.data[i][j]= x/r*vth + noise(rng);

            flow.p.data[i][j]=flow.rho.data[i][j]*cs*cs;
            double ke=0.5*flow.rho.data[i][j]*(flow.u.data[i][j]*flow.u.data[i][j]+flow.v.data[i][j]*flow.v.data[i][j]);
            flow.e.data[i][j]=flow.p.data[i][j]/(gamma-1.0)+ke;

            flow.bx.data[i][j]=0.0;
            flow.by.data[i][j]=0.01;
            flow.psi.data[i][j]=0.0;
        }
}
// new part for physics.cpp
void add_divergence_error(FlowField& flow, double amplitude) {
    // Add artificial divergence to test GLM
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < flow.bx.nx-1; ++i) {
        for (int j = 1; j < flow.bx.ny-1; ++j) {
            double x = flow.bx.x0 + i * flow.bx.dx - 0.5;
            double y = flow.bx.y0 + j * flow.bx.dy - 0.5;
            
            // Add divergent perturbation
            flow.bx.data[i][j] += amplitude * x * exp(-(x*x + y*y)/0.1);
            flow.by.data[i][j] += amplitude * y * exp(-(x*x + y*y)/0.1);
        }
    }
}

// Simple analytic magnetic field for testing divergence cleaning
void initialize_test_field(FlowField& flow) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < flow.bx.nx; ++i) {
        for (int j = 0; j < flow.bx.ny; ++j) {
            double x = flow.bx.x0 + i * flow.bx.dx - 0.5;
            double y = flow.bx.y0 + j * flow.bx.dy - 0.5;

            // Divergent magnetic field: curl-free radial pattern
            flow.bx.data[i][j] = x * std::exp(-(x*x + y*y));
            flow.by.data[i][j] = y * std::exp(-(x*x + y*y));

            // Initialize other fields trivially
            flow.rho.data[i][j] = 1.0;
            flow.u.data[i][j]   = 0.0;
            flow.v.data[i][j]   = 0.0;
            flow.p.data[i][j]   = 1.0;
            flow.e.data[i][j]   = 1.0/(1.4 - 1.0);
            flow.psi.data[i][j] = 0.0;
        }
    }
}

// Peak Bx test from "Hyperbolic Divergence Cleaning for the MHD Equations"
// Uniform fluid with a localized magnetic perturbation creating non-zero
// divergence initially.
void initialize_peak_bx(FlowField& flow) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < flow.bx.nx; ++i) {
        for (int j = 0; j < flow.bx.ny; ++j) {
            double x = flow.bx.x0 + i * flow.bx.dx - 0.5;
            double y = flow.bx.y0 + j * flow.bx.dy - 0.5;

            flow.rho.data[i][j] = 1.0;           // constant density
            flow.u.data[i][j]   = 0.0;           // initially at rest
            flow.v.data[i][j]   = 0.0;
            flow.p.data[i][j]   = 1.0;           // constant pressure
            flow.e.data[i][j]   = 1.0/(1.4 - 1.0);

            // Magnetic field: By uniform, Bx localized Gaussian peak
            double r2 = x*x + y*y;
            double peak = std::exp(-r2 / 0.01);  // narrow Gaussian
            flow.bx.data[i][j] = peak;           // localized divergence source
            flow.by.data[i][j] = 0.0;

            flow.psi.data[i][j] = 0.0;
        }
    }
}
