#include <iostream>
#include <vector>
#include <mpi.h>
#include <field_structures.h>
#include <ns3d_func.h>

// --------------------------- Simple demo main (initialization only) -----------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CartDecomp C;
    SolverParams P;
    // initialize SolverParams
    initialize_SolverParams(P, C);
    build_cart_decomp(C);

    GridDesc G; G.global_nx = 16; G.global_ny = 16; G.global_nz = 16; G.dx = G.dy = G.dz = 1.0/16.0;

    int ghost_layers = 3;
    LocalDesc L; compute_local_desc(G, C, L, /*ghosts*/ghost_layers,ghost_layers,ghost_layers);

    Field3D F; F.allocate(L);

    // Initialize field
    initialize_uniform_field(F, G, P);

    // exchange halos once
    HaloRequests reqs;
    exchange_halos_conserved(F, C, L, reqs);

    F.conservedToPrimitive(P); // update primitive variables (including ghosts)

    if (C.rank == 0) std::cout << "Initialization + halo exchange done\n";

    // output the initial field
    std::string prefix = "initial_field";
    write_tecplot_field(F, G, C, prefix, 0.0);

    // Main time-stepping loop
    int max_steps = 1000;
    int monitor_freq = 100;
    int output_freq = 200;
    double TotalTime = 10.0;
    time_advance(F, C, G, P, reqs, max_steps, monitor_freq, output_freq, TotalTime);

    // output the final field
    prefix = "final_field";
    write_tecplot_field(F, G, C, prefix, TotalTime);
    MPI_Finalize();
    return 0;
}