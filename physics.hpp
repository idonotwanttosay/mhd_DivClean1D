#pragma once
#include "grid.hpp"
void initialize_MHD_disk(FlowField& flow, int seed = 12345);
void add_divergence_error(FlowField& flow, double amplitude = 0.1);
// Initialize a simple magnetic field test function for divergence cleaning
void initialize_test_field(FlowField& flow);
// Initial condition inspired by "Hyperbolic Divergence Cleaning for the MHD Equations"
// Peak Bx problem: uniform state with a localized magnetic perturbation
void initialize_peak_bx(FlowField& flow);
