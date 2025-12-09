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
    int monitor_Stepfreq = P.monitor_Stepfreq;
    int output_Timefreq = P.output_Timefreq;
    double TotalTime = P.TotalTime;
    bool output_diagnostics = false;
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
        if (current_time + dt > output_Timefreq) {
            dt = output_Timefreq - current_time; // adjust to hit output time
            output_Timefreq += P.output_Timefreq;
            output_diagnostics = true;
        }
        current_time += dt;
        
        // 三阶Runge–Kutta推进
        runge_kutta_3(F, C, G, P, out_reqs, dt);
        if(C.rank == 0){
            std::cout << "Completed step " << step << " / " << max_steps << ", Time = " << current_time << " / " << TotalTime << "\n";
        }

        // 诊断与监控
        if (step % monitor_Stepfreq == 0 || step == 1) {
            const bool need_diag = P.monitor_res || P.monitor_energy || P.monitor_time_step || P.monitor_current_time;
            if (need_diag) {
                compute_diagnostics(F, P, G);
            }

            double t_now = MPI_Wtime();
            double t_elapsed = t_now - t_start;
            double t_step = t_now - t_last;
            t_last = t_now;

            if (C.rank == 0) {
                std::cout << std::fixed << std::setprecision(6) << "[Step " << step << "] ";
                if (P.monitor_time_step) {
                    std::cout << "dt=" << dt << "  Time/step=" << t_step << "s  Elapsed=" << t_elapsed << "s";
                }
                if (P.monitor_current_time) {
                    std::cout << "  Time=" << current_time << "/" << TotalTime;
                }
                if (P.monitor_energy) {
                    std::cout << "  E_avg=" << F.global_Etot;
                }
                std::cout << "\n";

                if (P.monitor_res) {
                    std::stringstream ss;
                    ss << "output_residuals.dat";
                    write_residuals_tecplot(F, step, ss.str());
                }
            }
        }

        // 输出流场文件
        if (output_diagnostics) {
            std::stringstream ss;
            ss << "output_time_" << std::setw(5) << std::setfill('0') << current_time;
            write_tecplot_field(F, G, C, ss.str(), current_time);
            if (P.isotropic_analyse) {
                std::stringstream ss;
                ss << "output_time_" << std::setw(5) << std::setfill('0') << current_time;
                isotropic_post_process(F, G, C, ss.str() + "_spectrum.dat");
            }
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