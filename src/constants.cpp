#include "constants.h"


const int Nx = 60;
const int Ny = 40;
const double omega = 1.5;

const int c[9][2] = {
  { 0,  0},
  { 1,  0},
  { 0,  1},
  {-1,  0},
  { 0, -1},
  { 1,  1},
  {-1,  1},
  {-1, -1},
  { 1, -1}
};

const double w[9] = {
  4.0/9.0,
  1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
