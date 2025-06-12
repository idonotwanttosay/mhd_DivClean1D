#pragma once
#include <string>
#include "grid.hpp"

void save_flow_MHD(const FlowField& flow, const std::string& dir, int step);
void save_amr_grid(const AMRGrid& amr, const std::string& dir, int step);
