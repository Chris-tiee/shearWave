#include <Kokkos_Core.hpp>
#include <iostream>
#include "utilities.h"
#include "constants.h"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>



int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    
    // Initialize density and speed
    Kokkos::View<double**> rho("rho", Nx, Ny);
    Kokkos::View<double**[2]> u("u", Nx, Ny);

    const double umax = 0.1;
    const int n = 1;
    const double pi = M_PI;
    const double k = 2.0 * pi * n / Ny;

    Kokkos::parallel_for("init_rho_u",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx, Ny}),
    KOKKOS_LAMBDA(int i, int j) {
        rho(i, j) = 1.0;
        // u(i, j, 0) = umax * sin(2*k * j);
        u(i, j, 0) = umax * sin(2*k * (j+0.5)); 
        u(i, j, 1) = 0.0;
    });

    
    // Initialize f
    Kokkos::View<double**[9]> f("f", Nx, Ny);
    f = equilibrium(rho, u);

    
    auto start = std::chrono::high_resolution_clock::now();
    
    //this gives us the amplitude of u at N/2, N/8
    std::ofstream ux_amp("ux.csv", std::ios::app);
    ux_amp  << u(Nx/2,Ny/8,0) <<",";
    ux_amp.close();
    
    std::ofstream ux_out("uxOut.csv", std::ios::app);
    for (int j = 0; j<Ny-1; j++){
      ux_out  << u(Nx/2,j,0) <<",";
    }
    ux_out << u(Nx/2,Ny-1,0)<<"\n";
    ux_out.close();
    
    
    Kokkos::View<double*> uNx("u",Ny);
    Kokkos::View<double*> rhoX("rho",Ny);

    for (int i = 1; i < 10001; ++i) {
      f = Streaming(f);
      f = Collision(f);
      //save in csvv f 
      if (i % 100 == 0) {
        Kokkos::parallel_for("compute_feq", Ny,
          KOKKOS_LAMBDA(int y) {
            int x = Nx/2;
            double ux = 0.0;
            double rho_xy = 0.0;

            for (int q = 0; q < 9; ++q) {
              ux += c[q][0]*f(x,y,q);
              rho_xy += f(x,y,q);
            }
            uNx(y) = ux/rho_xy;
            rhoX(y) =rho_xy;
        
        }); 

        std::ofstream ux_out("uxOut.csv", std::ios::app);
        for (int j = 0; j<Ny-1; j++){
          ux_out  << uNx(j) <<",";
        }
        ux_out << uNx(Ny-1)<<"\n";
        ux_out.close();

        std::ofstream rho_out("rho.csv", std::ios::app);
        for (int j = 0; j<Ny-1; j++){
          rho_out  << rhoX(j) <<",";
        }
        rho_out << rhoX(Ny-1)<<"\n";
        rho_out.close();

        // amplitude
        std::ofstream ux_amp("ux.csv", std::ios::app);
        ux_amp  << uNx(Ny/8) << (i < 9999 ? "," : "\n");
        ux_amp.close();


      }

    };

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

  }

  Kokkos::finalize();
  return 0;
}


 



    // // This is just to print u to see it
    // auto u_h = Kokkos::create_mirror_view(u);
    // Kokkos::deep_copy(u_h, u);

    // for (int i = 0; i < Nx; ++i) {
    //   for (int j = 0; j < Ny; ++j) {
    //     for (int q = 0; q < 1; ++q){
    //       std::cout << "u(" << i << "," << j << "," << q << ") = " << u_h(i,j,q) << "\n";
    //   }}
    // }

    // // This is just to print f to see it
    // auto f_h = Kokkos::create_mirror_view(f);
    // Kokkos::deep_copy(f_h, f);

    // for (int i = 0; i < Nx; ++i) {
    //   for (int j = 0; j < Ny; ++j) {
    //     for (int q = 1; q < 2; ++q){
    //       std::cout << "f(" << i << "," << j << "," << q << ") = " << f_h(i,j,q) << "\n";
    //   }}
    // }

    // // This is just to print rho to see it
    // auto rho_h = Kokkos::create_mirror_view(rho);
    // Kokkos::deep_copy(rho_h, rho);

    // for (int i = 0; i < Nx; ++i) {
    //   for (int j = 0; j < Ny; ++j) {
    //       std::cout << "rho(" << i << "," << j << "," <<  ") = " << rho_h(i,j) << "\n";
    // }};

    // //save in csvv f 
    // auto f_host = Kokkos::create_mirror_view(f);
    // Kokkos::deep_copy(f_host, f);
    // for (int q = 0; q < 9; ++q) {
    //   std::ofstream out("f_" + std::to_string(q) + ".csv");
    //   for (int i = 0; i < Nx; ++i) {
    //     for (int j = 0; j < Ny; ++j) {
    //       out << f_host(i, j, q);
    //       if (j < Ny - 1) out << ",";  // comma between columns
    //     }
    //     out << "\n";  // new row
    //   }
    //   out.close();
    // }
