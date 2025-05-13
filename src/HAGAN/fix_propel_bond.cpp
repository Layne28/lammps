/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Matthew S. E. Peterson (Brandeis University)

   Thanks to Stefan Paquay (Brandeis) and Abhijeet Joshi (Brandeis) for
   implementation help and useful advice!
------------------------------------------------------------------------- */

#include "fix_propel_bond.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fmt/core.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* -------------------------------------------------------------------------- */

FixPropelBond::FixPropelBond(LAMMPS * lmp, int narg, char **argv)
  : Fix(lmp, narg, argv)
  , magnitude{0.0}
  , reversal_time{0.0}
  , reversal_prob{0.0}
  , nmolecules{0}
  , ncheck{0}
  , nevery{0}
  , reversal_mode{OFF}
  , btype_flag{nullptr}
  , reverse{nullptr}
  , random{nullptr}
{
  if (narg < 4) {
    error->all(FLERR, "Illegal fix propel/bond command");
  }

  magnitude = utils::numeric(FLERR, argv[3], false, lmp);
  parse_keywords(narg - 4, argv + 4);
}

/* -------------------------------------------------------------------------- */

FixPropelBond::~FixPropelBond()
{
  if (random) delete random;
  memory->destroy(reverse);
  memory->destroy(btype_flag);
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::init()
{
  if (reversal_mode == ON) {
    grow_reversal_list();
    reversal_prob = update->dt / reversal_time;
  }
}

/* -------------------------------------------------------------------------- */

int FixPropelBond::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* -------------------------------------------------------------------------- */

double FixPropelBond::memory_usage()
{
  double bytes = sizeof(FixPropelBond);
  if (btype_flag) bytes += (atom->nbondtypes + 1.0) * sizeof(int);
  if (reverse) bytes += (nmolecules + 1.0) * sizeof(int);
  if (random) bytes += sizeof(RanMars);
  return bytes;
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::pre_force(int /* vlag */)
{
  int i, j, type, sign;
  double dx, dy, dz, r, scale;
  tagint mol;

  int newton_bond = force->newton_bond;
  
  int nbonds = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;

  for (int n = 0; n < nbonds; ++n) {
    i = bondlist[n][0];
    j = bondlist[n][1];
    type = bondlist[n][2];

    if (mask[i] & mask[j] & groupbit) {
      sign = (atom->tag[i] > atom->tag[j]) ? 1 : -1;
      sign *= btype_flag[type];
      
      if (reversal_mode == ON) {
        mol = atom->molecule[i];
        if (mol == atom->molecule[j] && mol <= nmolecules) {
          sign *= reverse[mol];
        } else {
          sign = 0;
        }
      }

      if (sign == 0) continue;

      dx = x[i][0] - x[j][0];
      dy = x[i][1] - x[j][1];
      dz = x[i][2] - x[j][2];
      r = std::sqrt(dx*dx + dy*dy + dz*dz);

      if (r > 0.0) scale = 0.5 * sign * magnitude / r;
      else scale = 0.0;

      if (newton_bond || i < nlocal) {
        f[i][0] += scale * dx;
        f[i][1] += scale * dy;
        f[i][2] += scale * dz;
      }

      if (newton_bond || j < nlocal) {
        f[j][0] += scale * dx;
        f[j][1] += scale * dy;
        f[j][2] += scale * dz;
      }
    }
  }

  if (reversal_mode == ON) update_reversal_time();
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::grow_reversal_list()
{
  tagint max = 0;
  for (int i = 0; i < atom->nlocal; ++i) {
    max = std::max(max, atom->molecule[i]);
  }

  {
    tagint max_all;
    MPI_Allreduce(&max, &max_all, 1, MPI_LMP_TAGINT, MPI_MAX, MPI_COMM_WORLD);
    max = max_all;
  }

  if (max > nmolecules) {
    reverse = memory->grow(reverse, max + 1, "propel/bond:reverse");
    for (int mol = nmolecules + 1; mol <= max; ++mol) {
      reverse[mol] = 1;
    }
    
    if (nmolecules == 0) reverse[0] = 1;
    nmolecules = max;
  }
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::update_reversal_time()
{
  if (nevery > 0) {
    if (--ncheck == 0) {
      grow_reversal_list();
      ncheck = nevery;
    }
  }

  // reversal times are implicitly sampled from a geometric distribution, the
  // discrete analog of the exponential distribution, since it is memoryless
  for (int i = 1; i <= nmolecules; ++i) {
    if (random->uniform() <= reversal_prob) reverse[i] = -reverse[i];
  }
}

/* -------------------------------------------------------------------------- */

void FixPropelBond::parse_keywords(int narg, char **argv)
{
  char **btype_argv = nullptr;
  int btype_narg = 0;
  
  char **flip_argv = nullptr;
  int flip_narg = 0;
  
  char **reverses_argv = nullptr;
  int reverses_narg = 0;

  char **check_argv = nullptr;
  int check_narg = 0;

  int iarg = 0;
  char *arg = nullptr;
  while (iarg < narg) {
    arg = argv[iarg++];
    
    if (strcmp(arg, "btypes") == 0) {
      if (btype_argv) {
        error->all(FLERR,
            "Illegal fix propel/bond command"
            " - 'btypes' keyword used more than once"
        );
      }
      
      btype_argv = argv + iarg;
      btype_narg = 0;
      while (iarg < narg) {
        arg = argv[iarg];
        if (std::isdigit(arg[0]) || arg[0] == '*') {
          iarg++;
          btype_narg++;
        } else {
          break;
        }
      }
    
    } else if (strcmp(arg, "flip") == 0) {
      if (flip_argv) {
        error->all(FLERR,
            "Illegal fix propel/bond command"
            " - 'flip' keyword used more than once"
        );
      }

      flip_argv = argv + iarg;
      flip_narg = 0;
      while (iarg < narg) {
        arg = argv[iarg];
        if (std::isdigit(arg[0]) || arg[0] == '*') {
          iarg++;
          flip_narg++;
        } else {
          break;
        }
      }

    } else if (strcmp(arg, "reverses") == 0) {
      if (reverses_argv) {
        error->all(FLERR,
            "Illegal fix propel/bond command"
            " - 'reverses' keyword used more than once"
        );
      }

      reverses_argv = argv + iarg;
      reverses_narg = 0;
      if (iarg + 2 <= narg) {
        iarg += 2;
        reverses_narg += 2;
      }

    } else if (strcmp(arg, "check") == 0) {
      if (check_argv) {
        error->all(FLERR,
            "Illegal fix propel/bond command"
            " - 'check' keyword used more than once"
        );
      }
      
      check_argv = argv + iarg;
      check_narg = 0;
      if (iarg + 1 <= narg) {
        iarg++;
        check_narg++;
      }

    } else {
      error->all(FLERR, fmt::format(
          "Illegal fix propel/bond command"
          " - Unknown keyword '{}'", arg)
      );
    }
  }

  btype_flag = memory->create(
      btype_flag, atom->nbondtypes + 1, "fix propel/bond:btype_flag"
  );
  for (int i = 0; i <= atom->nbondtypes; ++i) {
    btype_flag[i] = (btype_argv == nullptr);
  }

  if (btype_argv) {
    if (btype_narg == 0) {
      error->all(FLERR,
          "Illegal fix propel/bond command"
          " - No types given for 'btypes' keyword"
      );
    }

    int min = 1;
    int max = atom->nbondtypes;
    int ilo = 0;
    int ihi = 0;
    char *arg = nullptr;
    for (int i = 0; i < btype_narg; ++i) {
      arg = btype_argv[i];
      utils::bounds(FLERR, arg, min, max, ilo, ihi, error);
      for (int type = ilo; type <= ihi; ++type) {
        btype_flag[ilo] = 1;
      }
    }
  }

  if (flip_argv) {
    if (flip_narg == 0) {
      error->all(FLERR,
          "Illegal fix propel/bond command"
          " - No types given for 'flip' keyword"
      );
    }

    int min = 1;
    int max = atom->nbondtypes;
    int ilo = 0;
    int ihi = 0;
    char *arg = nullptr;
    for (int i = 0; i < flip_narg; ++i) {
      arg = flip_argv[i];
      utils::bounds(FLERR, arg, min, max, ilo, ihi, error);
      for (int type = ilo; type <= ihi; ++type) {
        btype_flag[type] = -btype_flag[type];
      }
    }
  }

  if (reverses_argv) {
    if (reverses_narg != 2) {
      error->all(FLERR,
          "Illegal fix propel/bond command"
          " - Incorrect number of values for keyword 'reverses'"
      );
    }

    reversal_mode = ON;
    reversal_time = utils::numeric(FLERR, reverses_argv[0], false, lmp);
    if (reversal_time <= 0.0) {
        error->all(FLERR,
            "Illegal fix propel/bond command"
            " - reversal time must be positive"
        );
      }
    
    int seed = utils::inumeric(FLERR, reverses_argv[1], false, lmp);
    random = new RanMars(lmp, seed);
  }

  if (check_argv) {
    if (check_narg != 1) {
      error->all(FLERR, 
          "Illegal fix propel/bond command"
          " - no value for keyword 'check'"
      );
     }

    if (reversal_mode == OFF) {
      error->warning(FLERR,
          "In fix propel/bond command"
          " - 'check' keyword will be ignored as 'reverses' keyword is unused"
      );
    }

    nevery = utils::inumeric(FLERR, check_argv[0], false, lmp);
    ncheck = nevery;
    if (nevery < 0) {
      error->all(FLERR, 
          "Illegal fix propel/bond command"
          " - 'check' value must be non-negative"
      );
    }
  }
}