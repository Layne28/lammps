/* -------atom-------------------atom--------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
   Contributed by Matthew S.E. Peterson @ Brandeis University

   Based on an initial implementation by Cong Qiao @ Brandeis.
   Thanks to Stefan Paquay @ Brandeis for help!

   Refactored and somewhat rewritten by Danny Hellstein @ Brandeis University
   Added interal datastuctures to keep track of dihedrals to reduce execution
time from N^2 -> N
------------------------------------------------------------------------- */

#include "fix_fluidize_mesh.h"

#include <math.h>
#include <stdlib.h>

#include <cmath>
#include <iostream>

#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "lammps.h"
#include "lmptype.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"
#define GROW_AMOUNT 4096

#define ENABLE_DEBUGGING 1

#define ERROR_ONE(...) error->one(FLERR, fmt::format(__VA_ARGS__))

#define ERROR_ALL(...) error->all(FLERR, fmt::format(__VA_ARGS__))

#define ILLEGAL(...) \
  error->all(FLERR,  \
             fmt::format("Illegal fix fluidize/mesh command - " __VA_ARGS__))

#if defined(ENABLE_DEBUGGING) && ENABLE_DEBUGGING == 1
#define DEBUG(...) fmt::print(__VA_ARGS__)
#else
#define DEBUG(...)
#endif

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

FixFluidizeMesh::FixFluidizeMesh(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg),
      random(nullptr),
      temperature(nullptr),
      swap_probability{},
      rmax2{},
      rmin2{},
      kbt{} {
  if (narg < 6 || narg > 12) {
    ILLEGAL("incorrect number of arguments");
  }
  arg += 3;
  narg -= 3;

  nevery = utils::inumeric(FLERR, arg[0], false, lmp);
  if (nevery <= 0) {
    ILLEGAL("nevery must be positive");
  }

  swap_probability = utils::numeric(FLERR, arg[1], false, lmp);
  if (swap_probability < 0 || swap_probability > 1) {
    ILLEGAL("swap probability must be in the range [0, 1]");
  }

  int seed = utils::inumeric(FLERR, arg[2], false, lmp);
  random = new RanMars(lmp, seed + comm->me);

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "rmax") == 0) {
      if (iarg + 1 >= narg) {
        ILLEGAL("no value given to keyword 'rmax'");
      }
      double rmax = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (rmax <= 0) {
        ILLEGAL("value of 'rmax' must be positive");
      }
      rmax2 = rmax * rmax;
      iarg += 2;
    } else if (strcmp(arg[iarg], "rmin") == 0) {
      if (iarg + 1 >= narg) {
        ILLEGAL("no value given to keyword 'rmin'");
      }
      double rmin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (rmin <= 0) {
        ILLEGAL("value of 'rmin' must be positive");
      }
      rmin2 = rmin * rmin;
      iarg += 2;
    } else if (strcmp(arg[iarg], "debug") ==0) {
      if (iarg +1 >= narg) ILLEGAL("no value given to keyword 'debug'");
      debug = (bool) utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      ILLEGAL("unknown keyword '{}'", arg[3]);
    }

  }

  if ((rmin2 > 0 || rmax2 > 0) && (rmin2 >= rmax2)) {
    ILLEGAL("Value of 'rmin' must be less than that of 'rmax'");
  }

  auto id_temp = id + std::string("_temp");
  modify->add_compute(id_temp + " all temp");

  int i = modify->find_compute(id_temp);
  if (i < 0) {
    ERROR_ALL("unable to compute system temperature");
  } else {
    temperature = modify->compute[i];
  }
}

/* ---------------------------------------------------------------------- */

FixFluidizeMesh::~FixFluidizeMesh() {
  if (random) delete random;
  if (temperature) modify->delete_compute(temperature->id);
}

/* ---------------------------------------------------------------------- */

int FixFluidizeMesh::setmask() {
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::init() {
  if (atom->molecular != Atom::MOLECULAR) {
    ERROR_ALL("fix fluidize/mesh requires a molecular system");
  }

  if (!force->newton_bond) {
    ERROR_ALL("fix fluidize/mesh requires 'newton on'");
  }

  if (!utils::strmatch(force->dihedral_style, "^harmonic")) {
    ERROR_ALL("fix fluidize/mesh requires harmonic dihedral style");
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::post_integrate() {
  if (update->ntimestep % nevery != 0) return;
  if (update->ntimestep % 10000 == 0) print_p_acc();

  comm->forward_comm();
  kbt = force->boltz * temperature->compute_scalar();

  bool skip;
  int a, b, c, d;
  int n_accept_old = n_accept;
  // construct a list of dihedrals and a corrsponding map from atom to all the
  // dihedrals it owns. this allows us random access into the full list of
  // dihedrals.
  int i = -1;
  int j = -1;
  int dihedral_cnt = 0;
  _atomToDihedral.resize(atom->natoms);
  for (auto &atom_it : _atomToDihedral) atom_it.clear();
  _connectivity.resize(atom->natoms);
  std::fill(_connectivity.begin(), _connectivity.end(), 0);
  _dihedralList.resize(atom->ndihedrals);


  for (int i_det = 0; i_det < atom->nlocal; ++i_det) {  // for each atom
    if (!(atom->mask[i_det] & groupbit))
      continue;  // if this fix doesn't apply to this type of atom
      // also update the connectivity for the bonds
      _connectivity[i_det] += atom->num_bond[i_det];
      for (int bond_num = 0; bond_num < atom->num_bond[i_det]; bond_num++){
        _connectivity[atom->map(atom->bond_atom[i_det][bond_num])]++; 
      }
    if (atom->num_dihedral[i_det] == 0)
      continue;  // if this atom doesn't have dihedrals continue

    for (int j_det = 0; j_det < atom->num_dihedral[i_det]; ++j_det) {  // for each dihedral on atom i_det
      if (atom->dihedral_atom2[i_det][j_det] != atom->tag[i_det])
        continue;  // I guess check if this diheardal exists?
      if (dihedral_cnt >= atom->ndihedrals)
        ERROR_ALL("double counting dihedrals");
      a = atom->map(atom->dihedral_atom1[i_det][j_det]);
      b = atom->map(atom->dihedral_atom2[i_det][j_det]);
      c = atom->map(atom->dihedral_atom3[i_det][j_det]);
      d = atom->map(atom->dihedral_atom4[i_det][j_det]);
      _dihedralList[dihedral_cnt] = {a, b, c, d};
      _dihedralList[dihedral_cnt].type = atom->dihedral_type[b][j_det];
      _dihedralList[dihedral_cnt].index = j_det;

      _atomToDihedral[b].insert(&_dihedralList[dihedral_cnt]);

      dihedral_cnt++;
    }


  }




  if (debug) {
    std::cout<<"Starting Connectivity Historgram:"<<std::endl;
      for (auto hist : count_vec_ele(_connectivity)){
        std::cout<<"{"<<hist.first<<", "<<hist.second<<"}"<<std::endl;
      }
    std::string name= "topology/flips_";
    name +=std::to_string(update->ntimestep);
    name +=".dump";
    file.open(name,std::fstream::out);
  }


  for (int count = 0; count < atom->ndihedrals; count++) {
    // Choose a dihedral at random; i = [0, ndihedral - 1]
    int index = random->integer(atom->ndihedrals);

    // This is like an attempt frequency -- sets overall rate
    if (random->uniform() > swap_probability) {
      continue;
    }

    a = _dihedralList[index].atoms[0];
    b = _dihedralList[index].atoms[1];
    c = _dihedralList[index].atoms[2];
    d = _dihedralList[index].atoms[3];

    skip = false;
    for (int id : {a, b, c, d}) {
      if (id < 0)
        ERROR_ONE("fix fluidize/mesh needs a larger communication cutoff!");

      if ((id >= atom->nlocal) || !(atom->mask[id] & groupbit)) {
        skip = true;
        break;
      }
    }
    // Make an attempt to flip the bond
    if (!skip) {
      if (check_candidacy(_dihedralList[index])) {
        try_swap(_dihedralList[index]);
      }
    }
  }
  if (debug) {
    file.close();
  }

  // std::cout << "Fluidization made " << n_accept - n_accept_old << " flips"
  //           << std::endl;
}

/* ---------------------------------------------------------------------- */

// Attempts the following transformation of a dihedral:
//
//    a               a
//   /               /|
//  b---c    -->    b | c
//     /              |/
//    d               d
// this transform implies
//    a        a
//   /          \  
//  b---c =  b---c
//     /      \    
//    d        d
//
// i.e. that {a b c d} is an equivalent dihedral to {a c b d}. We also need to
// modify the 4 dihedrals attached to this one
//
//  alpha ---- a ---- beta    alpha ---- a ---- beta
//       \    / \    /             \    /|\    /
//        \  1   2  /               \  1 | 2  /
//         \/     \/                 \/  |  \/
//          b -0- c       ===>        b  0  c
//         /\     /\                 /\  |  /\       
//        /  3   4  \               /  3 | 4  \      
//       /    \ /    \             /    \|/    \     
//  gamma ---- d ---- delta   gamma ---- d ---- delta
//
// alpha beta gamma and delta label the atoms exterior to this dihedral that
// must participate in this rearrangement 0 1 2 3 4 label the dihedrals that are
// modified when dihedral 0 is flipped
//
// we do not persevere any notion of handedness or parody in a dihedral, we find
// external dihedrals by center lines (or their reverse). This is why atom 1 (of
// {0,1,2,3}) 'owns' the dihedral

void FixFluidizeMesh::swap_dihedral(dihedral_type &dihedral) {
  // Performs all the actions to make a dihedral swap;
  tagint a, b, c, d;
  tagint *atoms = dihedral.atoms;
  a = atoms[0];
  b = atoms[1];
  c = atoms[2];
  d = atoms[3];

  // find the exterior dihedrals by their center bonds
  dihedral_type *exteriorDihedral1 = find_dihedral({b, a});
  dihedral_type *exteriorDihedral2 = find_dihedral({a, c});
  dihedral_type *exteriorDihedral3 = find_dihedral({c, d});
  dihedral_type *exteriorDihedral4 = find_dihedral({d, b});
  if (!(exteriorDihedral1 && exteriorDihedral2 && exteriorDihedral3 &&
        exteriorDihedral4))
    ERROR_ONE("couldn't find exterior dihedral while swapping back");
  // flip the central dihedral
  flip_central_dihedral(dihedral);

  // swap the appropriate atoms in the exterior dihedrals
  swap_atoms_in_dihedral(*exteriorDihedral1, c, d);
  swap_atoms_in_dihedral(*exteriorDihedral2, b, d);
  swap_atoms_in_dihedral(*exteriorDihedral3, b, a);
  swap_atoms_in_dihedral(*exteriorDihedral4, c, a);
}

/* ---------------------------------------------------------------------- */

double FixFluidizeMesh::swap_dihedral_calc_E(dihedral_type &dihedral) {
  // swaps a dihedral and returns the change in energy
  tagint a, b, c, d;
  tagint *atoms = dihedral.atoms;
  a = atoms[0];
  b = atoms[1];
  c = atoms[2];
  d = atoms[3];
  // find the exterior dihedrals by their center bonds
  dihedral_type *exteriorDihedral1 = find_dihedral({b, a});
  dihedral_type *exteriorDihedral2 = find_dihedral({a, c});
  dihedral_type *exteriorDihedral3 = find_dihedral({c, d});
  dihedral_type *exteriorDihedral4 = find_dihedral({d, b});
  if (!(exteriorDihedral1 && exteriorDihedral2 && exteriorDihedral3 &&
        exteriorDihedral4))
    ERROR_ONE("couldn't find exterior dihedral");

  old_n1 = *exteriorDihedral1;
  old_n2 = *exteriorDihedral2;
  old_n3 = *exteriorDihedral3;
  old_n4 = *exteriorDihedral4;

  // flip the central dihedral
  // find the inital energy of the dihedral
  double deltaE = -compute_bending_energy(dihedral);

  // find the inital energy of the bond
  bond_type centralBond = {b, c};
  find_bond(centralBond);
  if (!isinf(deltaE)) deltaE -= compute_bond_energy(centralBond);

  // flip the central bond update the bond and dihedral
  flip_central_dihedral(dihedral);

  // find the new bond energy
  centralBond = {a, d};
  find_bond(centralBond);
  if (!isinf(deltaE)) deltaE += compute_bond_energy(centralBond);

  // calculate the new dihedral energy
  if (!isinf(deltaE)) deltaE += compute_bending_energy(dihedral);

  // swap the appropriate atoms in the exterior dihedrals

  // compute their inital bending energy
  if (!isinf(deltaE)) deltaE -= compute_bending_energy(*exteriorDihedral1);
  if (!isinf(deltaE)) deltaE -= compute_bending_energy(*exteriorDihedral2);
  if (!isinf(deltaE)) deltaE -= compute_bending_energy(*exteriorDihedral3);
  if (!isinf(deltaE)) deltaE -= compute_bending_energy(*exteriorDihedral4);

  // make the subsitutions marked in the comment block above
  swap_atoms_in_dihedral(*exteriorDihedral1, c, d);
  swap_atoms_in_dihedral(*exteriorDihedral2, b, d);
  swap_atoms_in_dihedral(*exteriorDihedral3, b, a);
  swap_atoms_in_dihedral(*exteriorDihedral4, c, a);

  // compute the final energy
  if (!isinf(deltaE)) deltaE += compute_bending_energy(*exteriorDihedral1);
  if (!isinf(deltaE)) deltaE += compute_bending_energy(*exteriorDihedral2);
  if (!isinf(deltaE)) deltaE += compute_bending_energy(*exteriorDihedral3);
  if (!isinf(deltaE)) deltaE += compute_bending_energy(*exteriorDihedral4);

  return deltaE;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::flip_central_dihedral(dihedral_type &dihedral) {
  // flips the central bond of this dihedral and update the dihedral
  tagint a, b, c, d;
  a = dihedral.atoms[0];
  b = dihedral.atoms[1];
  c = dihedral.atoms[2];
  d = dihedral.atoms[3];

  // flip the bond
  check_central_bond(dihedral);
  flip_central_bond(dihedral);
  // remove the dihedral;
  remove_dihedral(dihedral);
  // swap the atoms
  dihedral.atoms[0] = b;
  dihedral.atoms[1] = a;
  dihedral.atoms[2] = d;
  dihedral.atoms[3] = c;

  // add the dihedral;
  insert_dihedral(dihedral);
}

/* ---------------------------------------------------------------------- */
void FixFluidizeMesh::check_central_bond(dihedral_type dihedral) {
  // flips the bond of a dihedral
  tagint a, b, c, d;
  a = dihedral.atoms[0];
  b = dihedral.atoms[1];
  c = dihedral.atoms[2];
  d = dihedral.atoms[3];
  bond_type old_bond = {b, c};
  bond_type new_bond = {a, d};
  if (!find_bond(old_bond)) {
    std::cout << "Candidate Bonds for <" << atom->tag[b] << ", " << atom->tag[c]
              << ">:" << std::endl;
    for (int i = 0; i < atom->num_bond[b]; i++) {
      std::cout << "<" << atom->tag[b] << ", " << atom->bond_atom[b][i] << ">"
                << std::endl;
    }
    for (int i = 0; i < atom->num_bond[c]; i++) {
      std::cout << "<" << atom->tag[c] << ", " << atom->bond_atom[c][i] << ">"
                << std::endl;
    }
    std::cout << find_bond(old_bond) << " with index " << old_bond.index
              << std::endl;

    ERROR_ONE("Error: could not find bonds to swap! (in try_swap)");
  }
  if (find_bond(new_bond)) {
    std::cout << "Dihedral is {"<<a<<", "<<b<<", "<<c<<", "<<d<<"}"<<std::endl;
    std::cout << "Candidate Bonds for <" << atom->tag[a] << ", " << atom->tag[d]
              << "> ("
              << "<" << a << ", " << d << ">):" << std::endl;
    for (int i = 0; i < atom->num_bond[a]; i++) {
      std::cout << "<" << atom->tag[a] << ", " << atom->bond_atom[a][i] << "> "
                << "(<" << a << ", " << atom->map(atom->bond_atom[a][i]) << ">)"
                << std::endl;
    }
    for (int i = 0; i < atom->num_bond[d]; i++) {
      std::cout << "<" << atom->tag[d] << ", " << atom->bond_atom[d][i] << "> "
                << "(<" << d << ", " << atom->map(atom->bond_atom[d][i]) << ">)"
                << std::endl;
    }
    std::cout << find_bond(new_bond) << " with index " << new_bond.index
              << std::endl;

    ERROR_ONE("Error: found new bond before creation! (in try_swap)");
  }
  new_bond.type = old_bond.type;
}
void FixFluidizeMesh::flip_central_bond(dihedral_type dihedral) {
  // flips the bond of a dihedral
  tagint a, b, c, d;
  a = dihedral.atoms[0];
  b = dihedral.atoms[1];
  c = dihedral.atoms[2];
  d = dihedral.atoms[3];
  bond_type old_bond = {b, c};
  bond_type new_bond = {a, d};
  if (!find_bond(old_bond)) {
    ERROR_ONE("Error: could not find bonds to swap! (in try_swap)");
  }
  if (find_bond(new_bond)) {
    ERROR_ONE("Error: found new bond before creation! (in try_swap)");
  }
  new_bond.type = old_bond.type;
  remove_bond(old_bond);
  insert_bond(new_bond);
}

/* ---------------------------------------------------------------------- */

bool FixFluidizeMesh::check_bond_length(bond_type bond) {
  // Check that new bond has acceptable length
  int a1 = bond.atoms[0];
  int b1 = bond.atoms[1];
  double dx = atom->x[b1][0] - atom->x[a1][0];
  double dy = atom->x[b1][1] - atom->x[a1][1];
  double dz = atom->x[b1][2] - atom->x[a1][2];
  domain->minimum_image(dx, dy, dz);
  double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 < rmin2 || r2 > rmax2) {
    return false;
  }
  return true;
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::try_swap(dihedral_type &dihedral) {
  // make the swap and calculate the change of energy
  double deltaE = swap_dihedral_calc_E(dihedral);
  // check acceptance based on energy change
  if (!accept_change(deltaE)) {
    // if we reject swap back (no need to calculate energy here)
    swap_dihedral(dihedral);
    n_reject++;
    return;
  }
  n_accept++;
  next_reneighbor = update->ntimestep;
  if (debug) report_swap(dihedral,deltaE);
  // consistency check
  // for (auto &dihedral : _dihedralList) {
    // check_central_bond(dihedral);
  // }

}
/* ---------------------------------------------------------------------- */
// print acceptance probability
void FixFluidizeMesh::print_p_acc() {
  std::cout << "No. swaps accepted: " << n_accept << std::endl;
  std::cout << "No. swaps rejected: " << n_reject << std::endl;
  std::cout << "Acceptance ratio: " << (1.0 * n_accept) / (n_accept + n_reject)
            << std::endl;
}

/* ---------------------------------------------------------------------- */

double FixFluidizeMesh::compute_bending_energy(dihedral_type dihedral) {
  // Get positions of atoms in dihedral. Note: b and c are bonded atoms.
  int a = dihedral.atoms[0];
  int b = dihedral.atoms[1];
  int c = dihedral.atoms[2];
  int d = dihedral.atoms[3];

  double xa = atom->x[a][0];
  double ya = atom->x[a][1];
  double za = atom->x[a][2];

  double xb = atom->x[b][0];
  double yb = atom->x[b][1];
  double zb = atom->x[b][2];

  double xc = atom->x[c][0];
  double yc = atom->x[c][1];
  double zc = atom->x[c][2];

  double xd = atom->x[d][0];
  double yd = atom->x[d][1];
  double zd = atom->x[d][2];

  // Calculate angle theta which is the angle bw the two planes made by
  // {a,c,b} and {b,c,d}

  double dx_cb = xc - xb;
  double dy_cb = yc - yb;
  double dz_cb = zc - zb;

  double dx_ab = xa - xb;
  double dy_ab = ya - yb;
  double dz_ab = za - zb;

  double dx_dc = xd - xc;
  double dy_dc = yd - yc;
  double dz_dc = zd - zc;

  double N1_x = (dy_cb * dz_dc) - (dz_cb * dy_dc);
  double N1_y = (dz_cb * dx_dc) - (dx_cb * dz_dc);
  double N1_z = (dx_cb * dy_dc) - (dy_cb * dx_dc);

  double N2_x = (-dy_cb * dz_ab) - (-dz_cb * dy_ab);
  double N2_y = (-dz_cb * dx_ab) - (-dx_cb * dz_ab);
  double N2_z = (-dx_cb * dy_ab) - (-dy_cb * dx_ab);

  double norm_N1 = std::sqrt(N1_x * N1_x + N1_y * N1_y + N1_z * N1_z);
  double norm_N2 = std::sqrt(N2_x * N2_x + N2_y * N2_y + N2_z * N2_z);

  N1_x = N1_x / norm_N1;
  N1_y = N1_y / norm_N1;
  N1_z = N1_z / norm_N1;

  N2_x = N2_x / norm_N2;
  N2_y = N2_y / norm_N2;
  N2_z = N2_z / norm_N2;

  double costheta = N1_x * N2_x + N1_y * N2_y + N1_z * N2_z;

  // Returns the energy associated with the dihedral
  // we use -costheta because lammps defines 180 as flat
  double E = force->dihedral->single(dihedral.type, acos(-costheta));
  if (debug) {
    double stiffness = 500;
    double diff = abs(E - stiffness*(1-costheta));
    // if (diff/abs(E)<1){
    //   std::cout<<"I got " <<stiffness*(1-costheta)<<" Lammps got "<<E<<std::endl;
    //   double theta = acos(costheta);
    //   double E2 = force->dihedral->single(dihedral.type, theta-180);
    //   std::cout<<E2;
    //   ERROR_ALL("Bad Dihedral Energy");
    // }
  }

  return E;


}

/* ---------------------------------------------------------------------- */

double FixFluidizeMesh::compute_bond_energy(bond_type bond) {
  int a = bond.atoms[0];
  int b = bond.atoms[1];

  double dx = atom->x[b][0] - atom->x[a][0];
  double dy = atom->x[b][1] - atom->x[a][1];
  double dz = atom->x[b][2] - atom->x[a][2];
  domain->minimum_image(dx, dy, dz);

  double f;  // unused
  double r2 = dx * dx + dy * dy + dz * dz;

  double energy = 0.0;
  if (force->bond) {
    if ((rmax2 > 0 && r2 > rmax2) || (rmin2 > 0 && r2 < rmin2)) {
      energy += std::numeric_limits<double>::infinity();
    } else {
      energy += force->bond->single(bond.type, r2, a, b, f);
      
      if (debug) {
        // do my own energy calculation 
        double r = sqrt(r2);
        double rmin = sqrt(rmin2);
        double rmax = sqrt(rmax2);
        // hardcoded valuse for right now
        double delta = 0.5;
        double sigma = 10000;
        double leq = 1;
        double k = 100;
        double E = 0;
        double E_repulsive = 0;
        double E_attractive = 0;
        double E_spring = 0;
        if (r<rmin+delta) E_repulsive= sigma*exp(1.0/(r-(rmin+delta)))/(r-rmin);
        if (r>rmax-delta) E_attractive= sigma*exp(1.0/((rmax-delta)-r))/(rmax-r);
        E_spring =(r-leq)*(r-leq)*0.5*k;
        E = E_repulsive+E_attractive+E_spring;

        if ((abs(energy-E)/std::max(abs(energy),abs(E)))>1e-4){
          std::cout<<"I think it should be "<<E<<" Lammps says "<<energy<<std::endl;
          std::cout<<"R = "<<r<<std::endl;
          std::cout<<"E_repulsive = "<<E_repulsive<<std::endl;
          std::cout<<"E_attractive = "<<E_attractive<<std::endl;
          std::cout<<"E_spring = "<<E_spring<<std::endl;
           
          ERROR_ALL("Bad Energy Calculation");

        }

      }
    }
  }

  return energy;
}

/* ---------------------------------------------------------------------- */

bool FixFluidizeMesh::accept_change(double deltaE) {
  // delta = new energy - old energy

  return (deltaE<0 || random->uniform() < std::exp(-deltaE / kbt));

}

/* ---------------------------------------------------------------------- */

bool FixFluidizeMesh::find_bond(bond_type &bond) {
  auto find = [&](int a, int b) {
    bond.atoms[0] = a;
    bond.atoms[1] = b;
    tagint target = atom->tag[b];
    for (int i = 0; i < atom->num_bond[a]; ++i) {
      if (atom->map(atom->bond_atom[a][i]) == b) {
        bond.type = atom->bond_type[a][i];
        bond.index = i;
        return true;
      }
    }

    bond.index = -1;
    return false;
  };

  int a = bond.atoms[0];
  int b = bond.atoms[1];
  return find(a, b) || find(b, a);
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::remove_bond(bond_type bond) {
  int a = bond.atoms[0];
  int b = bond.atoms[1];
  int index = bond.index;
  int &num_bonds = atom->num_bond[a];
  int *bond_type = atom->bond_type[a];
  tagint *bond_atom = atom->bond_atom[a];

  if (index < 0 || num_bonds == 0) {
    ERROR_ONE("attempted to remove bond that does not exist");
  }
  _connectivity[a]--;
  _connectivity[b]--;
  num_bonds--;
  if (index != num_bonds) {
    bond_atom[index] = bond_atom[num_bonds];
    bond_type[index] = bond_type[num_bonds];
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::insert_bond(bond_type bond) {
  int a = bond.atoms[0];
  int b = bond.atoms[1];
  if (atom->tag[a] > atom->tag[b]) std::swap(a, b);

  int &num_bonds = atom->num_bond[a];
  if (num_bonds < atom->bond_per_atom) {
    atom->bond_atom[a][num_bonds] = atom->tag[b];
    atom->bond_type[a][num_bonds] = bond.type;
    num_bonds++;
    _connectivity[a]++;
    _connectivity[b]++;
  } else {
    ERROR_ONE(
        "No space for addition bonds - consider increasing "
        "extra/bond/per/atom");
  }
}

/* ---------------------------------------------------------------------- */

FixFluidizeMesh::dihedral_type *FixFluidizeMesh::find_dihedral(
    bond_type central_bond) {
  auto find = [&](int b, int c) {
    // this is a function that returns a bool if it finds the dihedral
    // given two atoms a and b

    // get the iterator to the set of dihedrals in a
    for (auto &candidateDihedral : _atomToDihedral[b]) {
      // consider it found if the center bond is the same
      if (b == candidateDihedral->atoms[1] &&
          c == candidateDihedral->atoms[2]) {
        return candidateDihedral;
      }
    }
    return (dihedral_type *)nullptr;
  };
  int b = central_bond.atoms[0];
  int c = central_bond.atoms[1];
  dihedral_type *found_dihedral = find(b, c);
  if (!found_dihedral) found_dihedral = find(c, b);
  // if (!found_dihedral) ERROR_ONE("Couldn't find dihedral");
  return found_dihedral;
}

/* ---------------------------------------------------------------------- */
void FixFluidizeMesh::remove_dihedral(dihedral_type &dihedral) {
  int owner_atom = dihedral.atoms[1];  // owner atom of dihedral
  int index = dihedral.index;

  // aliases
  int &num_dihedral = atom->num_dihedral[owner_atom];
  int *dtype = atom->dihedral_type[owner_atom];
  tagint *atom1 = atom->dihedral_atom1[owner_atom];
  tagint *atom2 = atom->dihedral_atom2[owner_atom];
  tagint *atom3 = atom->dihedral_atom3[owner_atom];
  tagint *atom4 = atom->dihedral_atom4[owner_atom];

  if (index < 0 || num_dihedral == 0 || index >= num_dihedral) {
    ERROR_ONE("attempted to remove dihedral that does not exist");
  }
  int erased = _atomToDihedral[owner_atom].erase(&dihedral);
  if (!erased) ERROR_ONE("Didn't find the dihedral to erase");

  num_dihedral--;

  int atoms_to_check[4] = {atom->map(atom1[index]), atom->map(atom2[index]),
                           atom->map(atom3[index]), atom->map(atom4[index])};
  bool we_good = true;
  for (int i = 0; i < 4; i++) {
    if (atoms_to_check[i] != dihedral.atoms[i]) we_good = false;
  }
  if (!we_good) {
    std::cout << "Trying to remove {" << dihedral.atoms[0] << ", "
              << dihedral.atoms[1] << ", " << dihedral.atoms[2] << ", "
              << dihedral.atoms[3] << "}"
              << " which should match (but doesn't) {"
              << atom->map(atom1[index]) << ", " << atom->map(atom2[index])
              << ", " << atom->map(atom3[index]) << ", "
              << atom->map(atom4[index]) << "}" << std::endl;
  }

  if (index != num_dihedral) {
    // if the dihedral we want to remove isn't the last one we
    // swap the last dihedral into this slot, this means we need to
    // update its index as well.
    dihedral_type *swappedDihedral = find_dihedral(
        {atom->map(atom2[num_dihedral]), atom->map(atom3[num_dihedral])});
    if (swappedDihedral) {
      swappedDihedral->index = index;
      swappedDihedral->type = dtype[index];
    } else {
      ERROR_ONE("Couldn't find dihedral to swap while removing one");
    }

    dtype[index] = dtype[num_dihedral];
    atom1[index] = atom1[num_dihedral];
    atom2[index] = atom2[num_dihedral];
    atom3[index] = atom3[num_dihedral];
    atom4[index] = atom4[num_dihedral];
    // find this swapped dihedral
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::insert_dihedral(dihedral_type &dihedral) {
  int a = dihedral.atoms[0];
  int b = dihedral.atoms[1];
  int c = dihedral.atoms[2];
  int d = dihedral.atoms[3];

  int &num_dihedral = atom->num_dihedral[b];
  if (num_dihedral < atom->dihedral_per_atom) {
    // dihedrals_type stores golbal id's while dihedral_atomN works with local
    // ids (tag)
    atom->dihedral_atom1[b][num_dihedral] = atom->tag[a];
    atom->dihedral_atom2[b][num_dihedral] = atom->tag[b];
    atom->dihedral_atom3[b][num_dihedral] = atom->tag[c];
    atom->dihedral_atom4[b][num_dihedral] = atom->tag[d];
    atom->dihedral_type[b][num_dihedral] = dihedral.type;
    dihedral.index = num_dihedral;
    num_dihedral++;
    _atomToDihedral[b].insert(&dihedral);
  } else {
    ERROR_ONE(
        "No space for addition dihedrals - consider increasing "
        "extra/dihedral/per/atom");
  }
}

/* ---------------------------------------------------------------------- */

void FixFluidizeMesh::swap_atoms_in_dihedral(dihedral_type &dihedral,
                                             tagint oldAtom, tagint newAtom) {
  // find which atom we are replacing
  int atom_index = -1;
  for (int i = 0; i < 4; i++) {
    if (oldAtom == dihedral.atoms[i]) {
      atom_index = i;
      break;
    }
  }
  if (atom_index == -1) {
    ERROR_ONE("Couldn't find atom in dihedral durring substitution");
  }

  switch (atom_index) {
    case 1:
      // if its the 1st atom then we have to full remove and reinsert it
      // because that atom owns this dihedral
      remove_dihedral(dihedral);
      dihedral.atoms[1] = newAtom;
      insert_dihedral(dihedral);
      break;

    // if we're changing a different one than we have to do less
    case 0:
      dihedral.atoms[0] = newAtom;
      atom->dihedral_atom1[dihedral.atoms[1]][dihedral.index] =
          atom->tag[newAtom];
      break;

    case 2:
      dihedral.atoms[2] = newAtom;
      atom->dihedral_atom3[dihedral.atoms[1]][dihedral.index] =
          atom->tag[newAtom];
      break;

    case 3:
      dihedral.atoms[3] = newAtom;
      atom->dihedral_atom4[dihedral.atoms[1]][dihedral.index] =
          atom->tag[newAtom];
      break;
  }
}
void FixFluidizeMesh::report_swap(dihedral_type dihedral,double E){
  auto tag_convert = [&](dihedral_type di){
    dihedral_type tag_di = {atom->tag[di.atoms[0]],atom->tag[di.atoms[1]],atom->tag[di.atoms[2]],atom->tag[di.atoms[3]]};
    return tag_di;
  };


  tagint a, b, c, d;
  tagint *atoms = dihedral.atoms;
  a = atoms[0];
  b = atoms[1];
  c = atoms[2];
  d = atoms[3];
  // output positions as well i guess?
  double xa = atom->x[a][0];
  double ya = atom->x[a][1];
  double za = atom->x[a][2];

  double xb = atom->x[b][0];
  double yb = atom->x[b][1];
  double zb = atom->x[b][2];

  double xc = atom->x[c][0];
  double yc = atom->x[c][1];
  double zc = atom->x[c][2];

  double xd = atom->x[d][0];
  double yd = atom->x[d][1];
  double zd = atom->x[d][2];

  // find the exterior dihedrals by their center bonds // rember a<->b & c<->d for finding these
  dihedral_type exteriorDihedral1 = tag_convert(*find_dihedral({a, b}));
  dihedral_type exteriorDihedral2 = tag_convert(*find_dihedral({b, d}));
  dihedral_type exteriorDihedral3 = tag_convert(*find_dihedral({d, c}));
  dihedral_type exteriorDihedral4 = tag_convert(*find_dihedral({c, a}));
  old_n1 = tag_convert(old_n1);
  old_n2 = tag_convert(old_n2);
  old_n3 = tag_convert(old_n3);
  old_n4 = tag_convert(old_n4);
  a = atom->tag[atoms[0]];
  b = atom->tag[atoms[1]];
  c = atom->tag[atoms[2]];
  d = atom->tag[atoms[3]];

  file<<"main: {"<<b<<", "<<a<<", "<<d<<", "<<c<<"} -> {"<<a<<", "<<b<<", "<<c<<", "<<d<<"};  Neighbors: "\
      <<"{"<<old_n1.atoms[0]<<", "<<old_n1.atoms[1]<<", "<<old_n1.atoms[2]<<", "<<old_n1.atoms[3]<<"} -> "\
      <<"{"<<exteriorDihedral1.atoms[0]<<", "<<exteriorDihedral1.atoms[1]<<", "<<exteriorDihedral1.atoms[2]<<", "<<exteriorDihedral1.atoms[3]<<"} "<<d<<"->"<<c<< ","\
      <<"{"<<old_n2.atoms[0]<<", "<<old_n2.atoms[1]<<", "<<old_n2.atoms[2]<<", "<<old_n2.atoms[3]<<"} -> "\
      <<"{"<<exteriorDihedral2.atoms[0]<<", "<<exteriorDihedral2.atoms[1]<<", "<<exteriorDihedral2.atoms[2]<<", "<<exteriorDihedral2.atoms[3]<<"} "<<a<<"->"<<c<< ","\
      <<"{"<<old_n3.atoms[0]<<", "<<old_n3.atoms[1]<<", "<<old_n3.atoms[2]<<", "<<old_n3.atoms[3]<<"} -> "\
      <<"{"<<exteriorDihedral3.atoms[0]<<", "<<exteriorDihedral3.atoms[1]<<", "<<exteriorDihedral3.atoms[2]<<", "<<exteriorDihedral3.atoms[3]<<"} "<<a<<"->"<<b<< ","\
      <<"{"<<old_n4.atoms[0]<<", "<<old_n4.atoms[1]<<", "<<old_n4.atoms[2]<<", "<<old_n4.atoms[3]<<"} -> "\
      <<"{"<<exteriorDihedral4.atoms[0]<<", "<<exteriorDihedral4.atoms[1]<<", "<<exteriorDihedral4.atoms[2]<<", "<<exteriorDihedral4.atoms[3]<<"} "<<d<<"->"<<b<<" Energy: "<<E<<std::endl;


  file<<"{<"<<xb<<", "<<yb<<", "<<zb<<">, "<<"<"<<xa<<", "<<ya<<", "<<za<<">, "\
      << "<"<<xd<<", "<<yd<<", "<<zd<<">, "<<"<"<<xc<<", "<<yc<<", "<<zc<<">}"<<std::endl;

}

bool FixFluidizeMesh::check_candidacy(dihedral_type dihedral){
  tagint a, b, c, d;
  tagint *atoms = dihedral.atoms;
  a = atoms[0];
  b = atoms[1];
  c = atoms[2];
  d = atoms[3];
  // check the connectivity on b and c, we want to avoid them becoming degenerate. It should not get less than 3, (this can cause degenercy of bonds)
  // if the new bond already exisits than this flip would make us degenerate. Skip

  
  bond_type new_bond = {a, d};\
  if (_connectivity[b]<5||_connectivity[c]<5) return false;

  if (find_bond(new_bond)) {
    return false;
    std::cout << "Trying to remove {" << b << ", "<< c << "}"
              << "and add {"<< a << ","<<d<<"}"
              << "however that already exists"<<std::endl
              << "connectivity on " << b << " is "<< _connectivity[b] << " and "
              << c << " is " << _connectivity[c]<<std::endl;
   
    
    
    
    //  ERROR_ONE("Cannot Find Bond");
  }

  // also keep anything from getting above 10 bonds.. thats absurd
  if (_connectivity[a]>8 || _connectivity[d]>8) return false;

  return check_bond_length({a,d});
}

std::vector<std::array<tagint,2>> FixFluidizeMesh::find_all_bonds_on_node(tagint node){
  std::vector<std::array<tagint,2>> bond_list;

  int map_int = atom->map(node);
  // start with this atom
  for (int bond_num = 0; bond_num<atom->num_bond[map_int]; bond_num++){
    std::array<tagint,2> bond = {node,atom->bond_atom[map_int][bond_num]};
    bond_list.push_back(bond);

  }

  for (int i_det = 0; i_det < atom->nlocal; ++i_det) {  // for each atom
    for (int bond_num = 0; bond_num < atom->num_bond[i_det]; bond_num++){
      if (atom->bond_atom[i_det][bond_num]==node){
        std::array<tagint,2> bond = {atom->tag[i_det],node};

        bond_list.push_back(bond);
      }
    }
  }

  return bond_list;
}

std::map<int,int> FixFluidizeMesh::count_vec_ele(std::vector<int> vec){
  std::map<int,int> hist;
  for (auto &ele:vec){
    auto entry = hist.find(ele);
    if (entry == hist.end()){
      hist[ele] = 1;
    } else {
      entry->second++;
    }
  }
  return hist;
}