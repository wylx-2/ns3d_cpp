#include <mpi.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "ns3d_func.h"
#include "field_structures.h"

// -----------------------------------------------------------------------------
// 主时间推进循环 with monitor & output
// -----------------------------------------------------------------------------
void time_advance(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P)
{
    double t_start = MPI_Wtime();
    double t_last = t_start;
    double current_time = 0.0;
    int max_steps = P.max_steps;
    int monitor_freq = P.monitor_freq;
    int output_freq = P.output_freq;
    double TotalTime = P.TotalTime;
    HaloRequests out_reqs;

    for (int step = 1; step <= max_steps; ++step)
    {
        // record conserved variables at current step
        F.recordConservedTo0();

        // 计算时间步长
        double dt = compute_timestep(F, G, P);
        if (current_time + dt > TotalTime) {
            dt = TotalTime - current_time; // adjust last step to hit TotalTime
        }
        current_time += dt;

        // 三阶Runge–Kutta推进
        runge_kutta_3(F, C, G, P, out_reqs, dt);

        // 诊断与监控
        if (step % monitor_freq == 0 || step == 1) {
            compute_diagnostics(F, P);

            double t_now = MPI_Wtime();
            double t_elapsed = t_now - t_start;
            double t_step = t_now - t_last;
            t_last = t_now;

            if (C.rank == 0) {
                std::cout << std::fixed << std::setprecision(6)
                          << "[Step " << step << "] "
                          << "dt=" << dt
                          << "  E_avg=" << F.global_Etot/(F.L.nx*F.L.ny*F.L.nz)
                          << "  Time/step=" << t_step << "s"
                          << "  Elapsed=" << t_elapsed << "s\n";
                std::stringstream ss;
                ss << "output_residuals.dat";
                write_residuals_tecplot(F, step, ss.str());
            }
        }

        // 输出流场文件
        if (step % output_freq == 0) {
            std::stringstream ss;
            ss << "output_step_" << std::setw(5) << std::setfill('0') << step;
            write_tecplot_field(F, G, C, ss.str(), current_time);
        }

        if (current_time >= TotalTime) {
            break; // reached total simulation time
        }
    }

    if (C.rank == 0) {
        double total_time = MPI_Wtime() - t_start;
        std::cout << "Simulation complete. Total time: " << total_time << " s\n";
    }
}