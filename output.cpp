// output
#include "field_structures.h"
#include "ns3d_func.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// tecplot 输出函数
// Write local (per-rank) field data to a Tecplot ASCII file.
// The file will contain a single structured ZONE with I=L.nx, J=L.ny, K=L.nz
// and point-packed data. Coordinates are cell centers computed from GridDesc.
void write_tecplot_field(const Field3D &F, const GridDesc &G, const CartDecomp &C, const std::string &prefix)
{
	const LocalDesc &L = F.L;
	int rank = C.rank;

	std::ostringstream ss;
	ss << prefix << "_rank" << rank << ".dat";
	std::ofstream ofs(ss.str());
	if (!ofs) {
		std::cerr << "Failed to open output file " << ss.str() << " for writing\n";
		return;
	}

	ofs << "TITLE = \"NS3D Field Rank " << rank << "\"\n";
	ofs << "VARIABLES = \"X\" \"Y\" \"Z\" "
		<< "\"rho\" \"rhou\" \"rhov\" \"rhow\" \"E\" "
		<< "\"u\" \"v\" \"w\" \"p\" \"T\"\n";

	// structured zone with local physical sizes
	ofs << "ZONE T=\"rank_" << rank << "\" I=" << L.nx << " J=" << L.ny << " K=" << L.nz
		<< " DATAPACKING=POINT\n";

	ofs << std::scientific << std::setprecision(8);

	// loop over physical cells in i (fast), j, k order
	for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
		for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
			for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
				int gid = F.I(i,j,k);
				int gi = L.ox + (i - L.ngx);
				int gj = L.oy + (j - L.ngy);
				int gk = L.oz + (k - L.ngz);
				double x = G.x0 + (gi + 0.5) * G.dx;
				double y = G.y0 + (gj + 0.5) * G.dy;
				double z = G.z0 + (gk + 0.5) * G.dz;

				double rho = F.rho[gid];
				double rhou = F.rhou[gid];
				double rhov = F.rhov[gid];
				double rhow = F.rhow[gid];
				double E = F.E[gid];
				double u = F.u[gid];
				double v = F.v[gid];
				double w = F.w[gid];
				double p = F.p[gid];
				double T = F.T[gid];

				ofs << x << " " << y << " " << z << " "
					<< rho << " " << rhou << " " << rhov << " " << rhow << " " << E << " "
					<< u << " " << v << " " << w << " " << p << " " << T << "\n";
			}
		}
	}

	ofs.close();
	std::cerr << "Wrote Tecplot file: " << ss.str() << " (rank " << rank << ")\n";
}

