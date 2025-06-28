#include "utilities.h"
#include <Kokkos_Core.hpp>
#include "constants.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>


// Equilibrium is for sure correct
Kokkos::View<double**[9]> equilibrium(Kokkos::View<double**> rho, Kokkos::View<double**[2]> u) {
    Kokkos::View<double**[9]> f("f", Nx, Ny);
    Kokkos::parallel_for("compute_feq",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
      KOKKOS_LAMBDA(int x, int y) {
        double rho_ij = rho(x, y);
        double ux = u(x, y, 0);
        double uy = u(x, y, 1);
        double usq = ux * ux + uy * uy;

        for (int q = 0; q < 9; ++q) {
          double cu = 3.0 * (c[q][0] * ux + c[q][1] * uy);
          f(x, y, q) = w[q] * rho_ij * (1.0 + cu * (1.0 + 0.5 * cu) - 1.5 * usq);
        }
    });
    return f;
}

// Stream is for sure correct
Kokkos::View<double**[9]> Streaming(Kokkos::View<double**[9]> f) {
  
  Kokkos::View<double***> f_new("f_new", Nx, Ny, 9);
  Kokkos::parallel_for("streaming",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, 9}),
    KOKKOS_LAMBDA(int x, int y, int q) {
      int xp = (x + c[q][0] + Nx) % Nx;
      int yp = (y + c[q][1] + Ny) % Ny;
      f_new(xp, yp, q) = f(x, y, q);
    }
  );
  
  return f_new;

}
// -------------------------------------------------------------------

Kokkos::View<double**[9]> Collision(Kokkos::View<double**[9]> f) {
    
    Kokkos::View<double**> rho("rho", Nx, Ny);
    Kokkos::View<double**[2]> u("u", Nx, Ny);
    Kokkos::View<double**[9]> feq("f", Nx, Ny);

    Kokkos::parallel_for("compute_feq",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
      KOKKOS_LAMBDA(int x, int y) {
        double rho_xy = 0.0;
        double ux = 0.0;
        double uy = 0.0;

        for (int q = 0; q < 9; ++q) {
          rho_xy += f(x,y,q);
          ux += c[q][0]*f(x,y,q);
          uy += c[q][1]*f(x,y,q);
        }
        rho(x,y)= rho_xy;
        if (rho_xy > 0){
          u(x,y,0) = ux/rho_xy;
          u(x,y,1) = uy/rho_xy;
        } else{
          u(x,y,0) = 0.0;
          u(x,y,1) = 0.0;
        }
    });

    // std::ofstream rho_out("rho.csv", std::ios::app);
    // rho_out  <<"New Batch"<<"\n";
    // for (int i = 0; i < Nx; ++i) {
    //   for (int j = 0; j < Ny; ++j) {
    //       rho_out << "rho(" << i << "," << j << ") = " << rho-+++(i,j) << "\n";
    // }};
    // rho_out.close();
        
    feq = equilibrium(rho,u);
    
    Kokkos::View<double***> f_new("f_new", Nx, Ny, 9);
    Kokkos::parallel_for("add_in_place",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {Nx, Ny, 9}),
      KOKKOS_LAMBDA(int i, int j, int q) {
        f_new(i, j, q) = f(i, j, q) + omega*(feq(i,j,q)-f(i,j,q));
      }
    );

    return f_new;
    
}

// void write_fields(Kokkos::View<double**> rho,
//                   Kokkos::View<double**[2]> u,
//                   const std::string& prefix) {

//   auto rho_host = Kokkos::create_mirror_view(rho);
//   auto u_host = Kokkos::create_mirror_view(u);
//   Kokkos::deep_copy(rho_host, rho);
//   Kokkos::deep_copy(u_host, u);

//   std::ofstream rho_out(prefix + "_rho.csv");
//   std::ofstream ux_out(prefix + "_ux.csv");
//   std::ofstream uy_out(prefix + "_uy.csv");

//   for (int i = 0; i < Nx; ++i) {
//     for (int j = 0; j < Ny; ++j) {
//       rho_out << rho_host(i, j) << (j < Ny - 1 ? "," : "\n");
//       ux_out  << u_host(i, j, 0) << (j < Ny - 1 ? "," : "\n");
//       uy_out  << u_host(i, j, 1) << (j < Ny - 1 ? "," : "\n");
//     }
//   }

//   rho_out.close();
//   ux_out.close();
//   uy_out.close();
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

// // This is just to print u to see it
// auto u_h = Kokkos::create_mirror_view(u);
// Kokkos::deep_copy(u_h, u);

// for (int i = 0; i < Nx; ++i) {
//   for (int j = 0; j < Ny; ++j) {
//     for (int q = 0; q < 1; ++q){
//       std::cout << "u(" << i << "," << j << "," << q << ") = " << u_h(i,j,q) << "\n";
//   }}
// }


    // std::ofstream rho_out("rho.csv", std::ios::app);
    // rho_out  << u(Nx/2,Ny/8,0) <<",";
    // _out.close();