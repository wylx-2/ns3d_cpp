#include <iostream>
#include <vector>
#include <mpi.h>
#include <field_structures.h>
#include <ns3d_func.h>

// --------------------------- Simple demo main (initialization only) -----------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // global info description
    CartDecomp C;
    SolverParams P;
    GridDesc G; 
    // initialize SolverParams
    initialize_SolverParams(P, C, G);
    build_cart_decomp(C);

    LocalDesc L; 
    compute_local_desc(G, C, L, P.ghost_layers, P.ghost_layers, P.ghost_layers);

    Field3D F; 
    F.allocate(L);
    initialize_uniform_field(F, G, P);  // Initialize field
    apply_boundary(F, G, C, P); // apply boundary conditions and holo exchange
    F.primitiveToConserved(P); // update primitive variables (including ghosts)
    if (C.rank == 0) std::cout << "Initialization + halo exchange done\n";

    // output the initial field
    std::string prefix = "initial_field";
    write_tecplot_field(F, G, C, prefix, 0.0);

    // Main time-stepping loop
    HaloRequests reqs;
    time_advance(F, C, G, P, reqs);

    // output the final field
    prefix = "final_field";
    write_tecplot_field(F, G, C, prefix, P.TotalTime);
    MPI_Finalize();
    return 0;
}