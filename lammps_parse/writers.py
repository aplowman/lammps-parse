"""`lammps_parse.writers.py`"""

import os
import numpy as np
from pathlib import Path

from lammps_parse.utils import format_arr

__all__ = [
    'write_lammps_atoms',
    'write_lammps_inputs',
]


def write_lammps_atoms(supercell, atom_sites, species, species_idx, dir_path, atom_style,
                       charges=None):
    """Write file defining atoms for a LAMMPS simulation, using the `atomic` atom style.

    Parameters
    ----------
    supercell : ndarray of shape (3, 3)
        Array of column vectors representing the edge vectors of the supercell.
    atom_sites : ndarray of shape (3, N)
        Array of column vectors representing the position of each atom.
    species : ndarray of str of shape (M, )
        The distinct species of the atoms.
    species_idx : list or ndarray of shape (N, )
        Maps each atom site to a given species in `species`.
    dir_path : str or Path
        Directory in which to generate the atoms file.
    atom_style : str ("atomic" or "full")
        Corresponds to the LAMMPs command `atom_style`. Determines which columns are
        necessary in writing the atom data.
    charges : list or ndarray of float of shape (M, ), optional
        The charge associated with each species type. Only used if `atom_style` is "full".

    Returns
    -------
    None

    Notes
    -----
    For `atom_style` "atomic", output columns of the body of the generated file are:
    `atom-ID`, `atom-type`, `x`, `y`, `z`. For `atom_style` "full", output columns are:
    `atom-ID`, `molecule-ID`, `atom-type`, `q` (charge), `x`, `y`, `z`.

    """

    dir_path = Path(dir_path)

    if isinstance(charges, list):
        charges = np.array(charges)

    # Validation
    all_atom_styles = ['full', 'atomic']
    if atom_style not in all_atom_styles:
        msg = 'Atom style: "{}" not understood; must be one of: {}.'
        raise ValueError(msg.format(atom_style, all_atom_styles))

    num_atoms = atom_sites.shape[1]
    num_atom_types = len(species)

    xhi = supercell[0, 0]
    yhi = supercell[1, 1]
    zhi = supercell[2, 2]
    xy = supercell[0, 1]
    xz = supercell[0, 2]
    yz = supercell[1, 2]

    atom_id = np.arange(1, num_atoms + 1)[:, None]
    atom_type = (species_idx + 1)[:, None]

    if atom_style == 'full':
        mol_id = np.zeros((num_atoms, 1), dtype=int)
        ch = charges[species_idx][:, None]

    atom_path = dir_path.joinpath('atoms.lammps')

    with atom_path.open('w', newline='') as handle:

        handle.write('Header\n\n')

        handle.write('{:d} atoms\n'.format(num_atoms))
        handle.write('{:d} atom types\n\n'.format(num_atom_types))

        handle.write('0.0 {:14.9f} xlo xhi\n'.format(xhi))
        handle.write('0.0 {:14.9f} ylo yhi\n'.format(yhi))
        handle.write('0.0 {:14.9f} zlo zhi\n'.format(zhi))
        handle.write('\n')

        tilt_str = '{:<24.15f} {:<24.15f} {:<24.15f} xy xz yz'
        handle.write(tilt_str.format(xy, xz, yz))
        handle.write('\n\n')

        handle.write('Atoms\n\n')

        if atom_style == 'atomic':
            fmt_sp = ['{:>8d}', '{:>5d}', '{:>24.15f}']
            arrays = [atom_id, atom_type, atom_sites.T]

        elif atom_style == 'full':
            fmt_sp = ['{:>8d}', '{:>5d}', '{:>5d}', '{:>15.10f}', '{:>24.15f}']
            arrays = [atom_id, mol_id, atom_type, ch, atom_sites.T]

        arr_fm = format_arr(arrays, format_spec=fmt_sp, col_delim=' ')
        handle.write(arr_fm)

    return atom_path


def write_lammps_inputs(supercell, atom_sites, species, species_idx, dir_path,
                        parameters, interactions, atoms_file, atom_style,
                        atom_constraints=None, cell_constraints=None, computes=None,
                        thermos_dt=1, dump_dt=1, charges=None):
    """Generate LAMMPS input files for energy minimisation of a 3D periodic supercell.

    Parameters
    ----------
    supercell : ndarray of shape (3, 3)
        Array of column vectors representing the edge vectors of the supercell.
    atom_sites : ndarray of shape (3, N)
        Array of column vectors representing the positions of each atom.
    species : ndarray of str of shape (M, )
        The distinct species of the atoms.
    species_idx : list or ndarray of shape (N, )
        Maps each atom site to a given species in `species`.
    dir_path : str or Path
        Directory in which to generate input files.
    atom_style : str ("atomic" or "full")
        Corresponds to the LAMMPs command `atom_style`. Determines which columns are
        necessary in writing the atom data.
    atom_constraints : dict, optional
        A dict with the following keys:
            fix_`mn`_idx : ndarray of dimension 1
                The atom indices whose `m` and `n` coordinates are to be fixed, where
                valid pairs of `mn` are (`xy`, `xz`, `yz`). By default, set to None.
                Indexing starts from 1!
            fix_xyz_idx : ndarray of dimension 1
                The atom indices whose `x`, `y` and `z` coordinates are to be fixed. By
                default, set to None. Indexing starts from 1!
    cell_constraints : dict, optional
        A dict with the following keys:
            lengths_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which supercell vectors
                are to remain equal to one another.
            angles_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which supercell angles
                are to remain equal to one another.
            fix_lengths : str
                Some combination of 'a', 'b' and 'c'. Represents which supercell vectors
                are to remain fixed.
            fix_angles : str
                Some combination of 'a', 'b' and 'c'. Represents which supercell angles
                are to remain fixed.
    computes : list of str, optional
        A list of quantities to compute during the simulation. If not specified, set to a
        list with elements: `pe/atom`, `displace/atom`, `voronoi/atom`.
    thermos_dt : int, optional
        After this number of timesteps, output thermodynamics to the log file.
    dump_dt : int, optional
        After this number of timesteps, output dump file containing atom positions.
    charges : list or ndarray of float of shape (M, ), optional
        The charge associated with each species type. Only used if `atom_style` is "full".

    Notes
    -----
    See [1] for an explanation of the LAMMPS input script syntax.

    References
    ----------
    [1] http://www.afs.enea.it/software/lammps/doc17/html/Section_commands.html

    TODO:
    -   Add stress compute
    -   Implement cell constraints, see: http://lammps.sandia.gov/doc/fix_box_relax.html

    """

    computes_info = {
        'pe/atom': {
            'name': 'peatom',
            'dims': 1,
            'fmt': ['%20.10f'],
        },
        'displace/atom': {
            'name': 'datom',
            'dims': 4,
            'fmt': ['%20.10f'] * 4,
        },
        'voronoi/atom': {
            'name': 'voratom',
            'dims': 2,
            'fmt': ['%20.10f', '%5.f'],
        },
    }

    dir_path = Path(dir_path)

    if computes is None:
        computes = ['pe/atom', 'displace/atom', 'voronoi/atom']

    # Validation
    for c in computes:
        if computes_info.get(c) is None:
            raise NotImplementedError(
                'Compute "{}" is not understood.'.format(c))

    # Write file defining atom positions:
    write_lammps_atoms(supercell, atom_sites, species, species_idx, dir_path, atom_style,
                       charges=charges)

    command_lns = list(parameters)
    command_lns.append('atom_style   {}'.format(atom_style))
    command_lns.append('')
    command_lns.append('read_data    {}'.format(atoms_file))
    command_lns.append('')
    command_lns += interactions

    # Cell constraints (cell is fixed by default)
    fix_lengths = cell_constraints.get('fix_lengths')
    fix_angles = cell_constraints.get('fix_angles')
    angles_eq = cell_constraints.get('angles_equal')
    lengths_eq = cell_constraints.get('lengths_equal')

    fix_count = 1

    # Define arguments for the LAMMPS `fix box/relax` command:
    relax_fp = ['fixedpoint 0.0 0.0 0.0']
    relax_A = ['x 0.0']
    relax_B = ['y 0.0', 'scalexy yes']
    relax_C = ['z 0.0', 'scaleyz yes', 'scalexz yes']
    relax_all = ['tri 0.0']
    relax_couple_xy = ['couple xy']
    relax_couple_xz = ['couple xz']
    relax_couple_yz = ['couple yz']
    relax_couple_xyz = ['couple xyz']

    cell_cnst = []

    if not (fix_angles == 'abc' and fix_lengths == 'abc'):

        cell_cnst.append('fix {:d} all box/relax'.format(fix_count))
        fix_count += 1

        if fix_angles == 'abc':
            if fix_lengths is None:
                cell_cnst.extend(relax_A + relax_B + relax_C)
            elif fix_lengths == 'bc':
                cell_cnst.extend(relax_A)
            elif fix_lengths == 'ac':
                cell_cnst.extend(relax_B)
            elif fix_lengths == 'ab':
                cell_cnst.extend(relax_C)
            elif fix_lengths == 'a':
                cell_cnst.extend(relax_B + relax_C)
            elif fix_lengths == 'b':
                cell_cnst.extend(relax_A + relax_C)
            elif fix_lengths == 'c':
                cell_cnst.extend(relax_A + relax_B)

        elif fix_angles is None:

            if fix_lengths is None:
                cell_cnst.extend(relax_all)
            else:
                msg = ('Relaxing supercell angles and fixing some or all supercell '
                       'lengths is not implemented in the LAMMPS input file writer.')
                raise NotImplementedError(msg)
        else:
            msg = ('Fixing only some supercell angles is not implemented in the LAMMPS '
                   'input file writer.')
            raise NotImplementedError(msg)

        cell_cnst += relax_fp

    cell_cnst_str = ' '.join(cell_cnst)

    fix_lns = []
    if cell_cnst_str is not '':
        fix_lns = [cell_cnst_str]

    # Atom constraints
    fix_xy_idx = atom_constraints.get('fix_xy_idx')
    fix_xz_idx = atom_constraints.get('fix_xz_idx')
    fix_yz_idx = atom_constraints.get('fix_yz_idx')
    fix_xyz_idx = atom_constraints.get('fix_xyz_idx')

    if fix_xy_idx is not None:

        nfxy = len(fix_xy_idx)
        if nfxy == atom_sites.shape[1]:
            fxy_grp = 'all'
        else:
            fxy_grp = 'fix_xy'
            fxy_grp_ln = 'group {} id '.format(fxy_grp)
            fxy_grp_ln += ('{:d} ' * nfxy).format(*(fix_xy_idx))
            fix_lns.append(fxy_grp_ln)

        fix_lns.append('fix {:d} {} setforce 0.0 0.0 NULL'.format(fix_count, fxy_grp))
        fix_count += 1

    if fix_xz_idx is not None:

        nfxz = len(fix_xz_idx)
        if nfxz == atom_sites.shape[1]:
            fxz_grp = 'all'
        else:
            fxz_grp = 'fix_xz'
            fxz_grp_ln = 'group {} id '.format(fxz_grp)
            fxz_grp_ln += ('{:d} ' * nfxz).format(*(fix_xz_idx))
            fix_lns.append(fxz_grp_ln)

        fix_lns.append('fix {:d} {} setforce 0.0 NULL 0.0'.format(fix_count, fxz_grp))
        fix_count += 1

    if fix_yz_idx is not None:

        nfyz = len(fix_yz_idx)
        if nfyz == atom_sites.shape[1]:
            fyz_grp = 'all'
        else:
            fyz_grp = 'fix_yz'
            fyz_grp_ln = 'group {} id '.format(fyz_grp)
            fyz_grp_ln += ('{:d} ' * nfyz).format(*(fix_yz_idx))
            fix_lns.append(fyz_grp_ln)

        fix_lns.append('fix {:d} {} setforce NULL 0.0 0.0'.format(fix_count, fyz_grp))
        fix_count += 1

    if fix_xyz_idx is not None:

        nfxyz = len(fix_xyz_idx)
        if nfxyz == atom_sites.shape[1]:
            fxyz_grp = 'all'
        else:
            fxyz_grp = 'fix_xyz'
            fxyz_grp_ln = 'group {} id '.format(fxyz_grp)
            fxyz_grp_ln += ('{:d} ' * nfxyz).format(*(fix_xyz_idx))
            fix_lns.append(fxyz_grp_ln)

        fix_lns.append('fix {:d} {} setforce 0.0 0.0 0.0'.format(fix_count, fxyz_grp))
        fix_count += 1

    # computes are used in the dump files
    dmp_computes = ''
    dmp_fmt = ''
    compute_lns = []
    for c in computes:

        cinf = computes_info[c]
        c_nm = cinf['name']
        c_dm = cinf['dims']
        c_fm = cinf['fmt']
        compute_lns.append('compute {} all {}'.format(c_nm, c))

        if c_dm == 1:
            dmp_computes += ' c_{}'.format(c_nm)
            dmp_fmt += ' {}'.format(c_fm[0])

        else:
            for i in range(c_dm):
                dmp_computes += ' c_{}['.format(c_nm) + '{:d}]'.format(i + 1)
                dmp_fmt += ' {}'.format(c_fm[i])

    # thermo prints info to the log file
    thermo_args = ['step', 'atoms', 'pe', 'ke', 'etotal', 'fmax']
    thermo_args_all = ' '.join(thermo_args)
    thermo_lns = [
        'thermo_style custom {}'.format(thermo_args_all),
        'thermo_modify format float %20.10f',
        'thermo {:d}'.format(thermos_dt)
    ]

    if atom_style == 'atomic':
        dump_str = 'dump 1 all custom {} dump.*.txt id type x y z'.format(dump_dt)
        dump_mod = 'dump_modify 1 format line "%5d %5d %20.10f %20.10f %20.10f'

    elif atom_style == 'full':
        dump_str = 'dump 1 all custom {} dump.*.txt id type x y z q'.format(dump_dt)
        dump_mod = 'dump_modify 1 format line "%5d %5d %20.10f %20.10f %20.10f %20.10f'

    dump_str += dmp_computes
    dump_mod += dmp_fmt + '"'
    dump_lns = [dump_str, dump_mod]

    # Run simulation

    # minimize args:
    #   energy_tol: energy change / energy magnitude
    #   force tol: ev / ang for units = metal
    #   max iterations:
    #   max force/energy evaluations:
    max_iters = int(1e4)
    max_force_per_energy_evals = int(1e5)
    sim_lns = []
    sim_lns.append('min_style cg')
    sim_lns.append('minimize 0.0 1.0e-6 {} {}'.format(
        str(max_iters),
        str(max_force_per_energy_evals)
    ))

    all_lns = [command_lns, fix_lns, compute_lns, thermo_lns, dump_lns, sim_lns]

    # Write all lines to input script file

    input_path = dir_path.joinpath('in.lammps')
    with input_path.open('w', newline='') as handle:
        for i_idx, i in enumerate(all_lns):
            if i_idx > 0:
                handle.write('\n')
            handle.writelines([j + '\n' for j in i])

    return input_path
