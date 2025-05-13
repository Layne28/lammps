/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(fluidize/mesh, FixFluidizeMesh)

#else

#ifndef LMP_FIX_FLUIDIZE_MESH_H
#define LMP_FIX_FLUIDIZE_MESH_H

#include <unordered_set>
#include <map>
#include <vector>
#include <fstream>

#include "fix.h"
namespace LAMMPS_NS {

class FixFluidizeMesh : public Fix {
struct bond_type {
  tagint atoms[2];
  int type;
  int index;
};

/* ---------------------------------------------------------------------- */

struct dihedral_type {
  tagint atoms[4];
  int type;
  int index;
};

 public:
  FixFluidizeMesh(class LAMMPS *, int, char **);
  virtual ~FixFluidizeMesh();

  int setmask() override;
  void init() override;
  void post_integrate() override;

 private:
  class RanMars *random;
  class Compute *temperature;
  double swap_probability;
  double rmax2;
  double rmin2;
  double kbt;

  dihedral_type old_dihedral;
  dihedral_type old_n1;
  dihedral_type old_n2;
  dihedral_type old_n3;
  dihedral_type old_n4;
  

  bool debug=false;
  std::fstream file;


  std::vector<std::unordered_set<dihedral_type *> > _atomToDihedral;
  std::vector<dihedral_type> _dihedralList;
  std::vector<int> _connectivity;

  int n_accept = 0;
  int n_reject = 0;

  double compute_bending_energy(dihedral_type);

  bool accept_change(double);

  void try_swap(dihedral_type &);
  void flip_central_dihedral(dihedral_type &dihderal);
  void flip_central_bond(dihedral_type dihedral);
  void swap_dihedral(dihedral_type &dihderal);
  double swap_dihedral_calc_E(dihedral_type &dihderal);

  void print_p_acc();

  bool find_bond(bond_type &);
  void remove_bond(bond_type);
  void insert_bond(bond_type);
  bool check_bond_length(bond_type);
  double compute_bond_energy(bond_type bond);

  dihedral_type *find_dihedral(bond_type centralBond);
  void remove_dihedral(dihedral_type &);
  void insert_dihedral(dihedral_type &);
  void swap_atoms_in_dihedral(dihedral_type &dihedral, tagint oldAtom,
                              tagint newAtom);

  int find_exterior_atom(dihedral_type a, dihedral_type b);

  void check_central_bond(dihedral_type dihedral);
  void report_swap(dihedral_type,double);
  bool check_candidacy(dihedral_type);
  void write_flip(dihedral_type dihedral);

  std::vector<std::array<tagint,2>> find_all_bonds_on_node(tagint node);
  std::map<int,int> count_vec_ele(std::vector<int> vec);
};

}  // namespace LAMMPS_NS

#endif
#endif

    /*
    ERROR/WARNING messages:

    E: Illegal ... command

    Self-explanatory.  Check the input script syntax and compare to the
    documentation for the command.  You can use -echo screen as a
    command-line option when running LAMMPS to see the offending line.

    */
