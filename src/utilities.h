#ifndef EQUILIBRIUM_H
#define EQUILIBRIUM_H
#include <Kokkos_Core.hpp>

// Declare equilibrium function
Kokkos::View<double**[9]> equilibrium(Kokkos::View<double**> rho,
                                        Kokkos::View<double**[2]> u);

Kokkos::View<double**[9]> Streaming(Kokkos::View<double**[9]> f); 
Kokkos::View<double**[9]> Collision(Kokkos::View<double**[9]> f);                                       
// void write_fields(Kokkos::View<double**> rho,
//                   Kokkos::View<double**[2]> u,
//                   const std::string& prefix);
 
#endif // __EQUILIBRIUM_H