#include <iostream>
#include <vector>
#include <mpi.h>
#include <field_structures.h>
#include <ns3d_func.h>

// --------------------------- Simple demo main (initialization only) -----------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CartDecomp C;
    // use MPI_Dims_create to choose a good px,py,pz
    int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int dims[3] = {0,0,0}; MPI_Dims_create(ranks, 3, dims);
    build_cart_decomp(C, dims[0], dims[1], dims[2], /*periodic*/true);

    GridDesc G; G.global_nx = 64; G.global_ny = 64; G.global_nz = 64; G.dx = G.dy = G.dz = 1.0/64.0;

    int ghost_layers = 3;
    LocalDesc L; compute_local_desc(G, C, L, /*ghosts*/ghost_layers,ghost_layers,ghost_layers);

    Field3D F; F.allocate(L);

    // init some field (e.g., constant density + small velocity perturbation in interior)
    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        F.rho[id] = 1.0;
        F.rhou[id] = 1.0;
        F.rhov[id] = 1.0;
        F.rhow[id] = 1.0;
        F.E[id] = 1.0;
    }



    // exchange halos once
    HaloRequests reqs;
    exchange_halos_conserved(F, C, L, reqs);

    SolverParams P;
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