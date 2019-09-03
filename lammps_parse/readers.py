"""`lammps_parse.readers.py`"""

from pathlib import Path

import numpy as np

__all__ = [
    'read_lammps_log',
    'read_lammps_dump',
    'read_lammps_output',
    'check_success',
]


def read_lammps_log(path):
    """Parse a Lammps log file.

    Parameters
    ----------
    path : str or Path
        File path to the Lammps log file to be read.

    Returns
    -------
    log_data : dict
        Dict with the following keys:
            version : str
            warnings : list
            errors : list
            thermo : dict
            dump_name : str

    """

    # Map `thermo_style` args to how they appears in the thermo output,
    # and their data type.
    thermo_map = {
        'step': {
            'name': 'Step',
            'dtype': int
        },
        'atoms': {
            'name': 'Atoms',
            'dtype': int
        },
        'pe': {
            'name': 'PotEng',
            'dtype': float
        },
        'ke': {
            'name': 'KinEng',
            'dtype': float
        },
        'etotal': {
            'name': 'TotEng',
            'dtype': float
        },
        'fmax': {
            'name': 'Fmax',
            'dtype': float,
        }
    }

    DUMP = 'dump'
    THERMO_STYLE = 'thermo_style'
    THERMO_OUT_END = 'Loop'
    WARN = 'WARNING:'
    ERR = 'ERROR:'
    VERS = 'LAMMPS'
    pot_file_match = 'pair_coeff'

    dump_name = None
    thermo_style_args = None
    thermo_style_out = None
    vers = None
    pot_file = None

    warns = []
    errs = []

    path = Path(path)

    with path.open('r', encoding='utf-8', newline='') as handle:

        mode = 'scan'
        for ln in handle:

            ln = ln.strip()
            ln_s = ln.split()

            if ln == '':
                continue

            if mode == 'scan':

                if pot_file_match in ln:
                    pot_file = pot_file_match.split('"')[1:2]

                if VERS in ln:
                    vers = ln.split('(')[1].split(')')[0]

                if ln_s[0] == DUMP:
                    dump_name = ln_s[5]

                elif ln_s[0] == THERMO_STYLE:
                    thermo_style_args = ln_s[2:]
                    tm = thermo_map
                    thermo_style_out = [tm[i]['name'] for i in thermo_style_args]
                    thermo_out = {i: [] for i in thermo_style_out}

                if thermo_style_out is not None:
                    if ln_s[0] == thermo_style_out[0]:
                        mode = 'thermo'
                        continue

                if WARN in ln:
                    warns.append(ln)

                if ERR in ln:
                    errs.append(ln)

            if mode == 'thermo':
                if ln_s[0] == THERMO_OUT_END:
                    mode = 'scan'
                else:
                    for i_idx, i in enumerate(thermo_style_out):
                        thermo_out[i].append(ln_s[i_idx])

    # Parse thermo as correct dtypes
    for k, v in thermo_out.items():
        dt = thermo_map[thermo_style_args[thermo_style_out.index(k)]]['dtype']
        thermo_out[k] = np.array(v, dtype=dt)

    log_data = {
        'version': vers,
        'warnings': warns,
        'errors': errs,
        'thermo': thermo_out,
        'dump_name': dump_name
    }

    return log_data


def read_lammps_dump(path):
    """Parse a Lammps dump file.

    Parameters
    ----------
    path : str or Path
        File path to the Lammps log file to be read.

    Returns
    -------
    dump_data : dict
        Dict with the following keys:
            time_step : int 
            num_atoms : int 
            box_tilt : bool
            box_periodicity : list of str
            box : ndarray
            supercell : ndarray
            atom_sites : ndarray
            atom_types : ndarray
            atom_pot_energy : ndarray
            atom_disp : ndarray
            vor_vols : ndarray
            vor_faces: ndarray

    Notes
    -----
    This is not generalised, in terms of the fields present in the `ATOMS` block, but
    could be developed to be more-generalised in the future.

    """

    # Search strings
    TS = 'ITEM: TIMESTEP'
    NUM_ATOMS = 'ITEM: NUMBER OF ATOMS'
    BOX = 'ITEM: BOX BOUNDS'
    ATOMS = 'ITEM: ATOMS'
    TILT_FACTORS = 'xy xz yz'

    ts = None
    num_atoms = None
    box_tilt = False
    box_periodicity = None
    box = []
    atom_sites = None
    atom_types = None
    atom_disp = None
    atom_pot = None
    vor_vols = None
    vor_faces = None

    with open(path, 'r', encoding='utf-8', newline='') as df:
        mode = 'scan'
        for ln in df:

            ln = ln.strip()
            ln_s = ln.split()

            if TS in ln:
                mode = 'ts'
                continue

            elif NUM_ATOMS in ln:
                mode = 'num_atoms'
                continue

            elif BOX in ln:
                mode = 'box'
                box_ln_idx = 0
                box_periodicity = [ln_s[-i] for i in [3, 2, 1]]
                if TILT_FACTORS in ln:
                    box_tilt = True
                continue

            elif ATOMS in ln:
                mode = 'atoms'
                headers = ln_s[2:]

                x_col = headers.index('x')
                y_col = headers.index('y')
                z_col = headers.index('z')

                atom_type_col = headers.index('type')
                vor_vol_col = headers.index('c_voratom[1]')
                vor_face_col = headers.index('c_voratom[2]')
                d1c = headers.index('c_datom[1]')
                d2c = headers.index('c_datom[2]')
                d3c = headers.index('c_datom[3]')
                d4c = headers.index('c_datom[4]')
                pot_col = headers.index('c_peatom')

                atom_ln_idx = 0
                atom_sites = np.zeros((3, num_atoms))
                atom_types = np.zeros((num_atoms,), dtype=int)
                atom_pot = np.zeros((num_atoms,))
                atom_disp = np.zeros((4, num_atoms))
                vor_vols = np.zeros((num_atoms,))
                vor_faces = np.zeros((num_atoms,), dtype=int)

                continue

            if mode == 'ts':
                ts = int(ln)
                mode = 'scan'

            elif mode == 'num_atoms':
                num_atoms = int(ln)
                mode = 'scan'

            elif mode == 'box':
                box.append([float(i) for i in ln_s])
                box_ln_idx += 1
                if box_ln_idx == 3:
                    mode = 'scan'

            elif mode == 'atoms':

                atom_sites[:, atom_ln_idx] = [
                    float(i) for i in (ln_s[x_col], ln_s[y_col], ln_s[z_col])]

                atom_disp[:, atom_ln_idx] = [
                    float(i) for i in [ln_s[j] for j in (d1c, d2c, d3c, d4c)]]

                atom_pot[atom_ln_idx] = float(ln_s[pot_col])
                atom_types[atom_ln_idx] = int(ln_s[atom_type_col])
                vor_vols[atom_ln_idx] = float(ln_s[vor_vol_col])
                vor_faces[atom_ln_idx] = int(ln_s[vor_face_col])

                atom_ln_idx += 1
                if atom_ln_idx == num_atoms:
                    mode = 'scan'

    # Form supercell edge vectors as column vectors:
    box = np.array(box)
    xlo_bnd = box[0, 0]
    xhi_bnd = box[0, 1]
    ylo_bnd = box[1, 0]
    yhi_bnd = box[1, 1]
    zlo_bnd = box[2, 0]
    zhi_bnd = box[2, 1]
    xy = box[0, 2]
    xz = box[1, 2]
    yz = box[2, 2]

    xlo = xlo_bnd - np.min([0, xy, xz, xy + xz])
    xhi = xhi_bnd - np.max([0, xy, xz, xy + xz])
    ylo = ylo_bnd - np.min([0, yz])
    yhi = yhi_bnd - np.max([0, yz])
    zlo = zlo_bnd
    zhi = zhi_bnd

    supercell = np.array([
        [xhi - xlo, xy, xz],
        [0, yhi - ylo, yz],
        [0, 0, zhi - zlo],
    ])

    dump_data = {
        'time_step': ts,
        'num_atoms': num_atoms,
        'box_tilt': box_tilt,
        'box_periodicity': box_periodicity,
        'box': box,
        'supercell': supercell,
        'atom_sites': atom_sites,
        'atom_types': atom_types,
        'atom_pot_energy': atom_pot,
        'atom_disp': atom_disp,
        'vor_vols': vor_vols,
        'vor_faces': vor_faces
    }

    return dump_data


def read_lammps_output(dir_path, log_name='log.lammps'):
    """Parse output files from a Lammps simulation.

    Parameters
    ----------
    dir_path : str or Path
        Directory in which to search for Lammps output files.
    log_name : str
        Name of the log file that exists in `dir_path`.

    Returns
    -------
    lammps_output : dict

    """

    dir_path = Path(dir_path)

    # Get the format of dump files from the log file
    log_path = dir_path.joinpath(log_name)
    log_data = read_lammps_log(log_path)

    all_dumps = {}
    atoms = []
    atom_disp = []
    atom_pot_energy = []
    vor_vols = []
    vor_faces = []
    supercell = []
    box = []
    time_steps = []

    for dump_file in dir_path.glob(log_data['dump_name']):

        dmp_i = read_lammps_dump(dump_file)
        dmp_ts = dmp_i['time_step']
        all_dumps.update({dmp_ts: dmp_i})

        atoms.append(dmp_i['atom_sites'])
        atom_disp.append(dmp_i['atom_disp'])
        atom_pot_energy.append(dmp_i['atom_pot_energy'])
        vor_vols.append(dmp_i['vor_vols'])
        vor_faces.append(dmp_i['vor_faces'])
        supercell.append(dmp_i['supercell'])
        box.append(dmp_i['box'])
        time_steps.append(dmp_i['time_step'])

    time_steps = np.array(time_steps)
    srt_idx = np.argsort(time_steps)
    atoms = np.array(atoms)[srt_idx]
    atom_disp = np.array(atom_disp)[srt_idx]
    atom_pot_energy = np.array(atom_pot_energy)[srt_idx]
    vor_vols = np.array(vor_vols)[srt_idx]
    vor_faces = np.array(vor_faces)[srt_idx]
    supercell = np.array(supercell)[srt_idx]
    box = np.array(box)[srt_idx]

    final_energy = log_data['thermo']['TotEng']

    lammps_output = {
        **log_data,
        'dumps': all_dumps,
        'atoms': atoms,
        'num_atoms': atoms[0].shape[1],
        'atom_disp': atom_disp,
        'atom_pot_energy': atom_pot_energy,
        'vor_vols': vor_vols,
        'vor_faces': vor_faces,
        'supercell': supercell,
        'box': box,
        'time_steps': time_steps,
        'final_energy': final_energy
    }

    return lammps_output


def check_success(dir_path, log_name='log.lammps'):
    """Check if the files in a given directory are indicative of a successful Lammps run.

    Parameters
    ----------
    dir_path : str or Path
        Directory in which to search for Lammps output files.
    log_name : str
        Name of the log file that exists in `dir_path`.

    Returns
    -------
    bool

    """

    dir_path = Path(dir_path)

    # Check log file exists:
    log_path = dir_path.joinpath(log_name)
    try:
        log_data = read_lammps_log(log_path)
    except FileNotFoundError:
        return False

    # Check no errors in log file:
    if log_data['errors']:
        return False

    # Check at least one dump file exists:
    if not dir_path.glob(log_data['dump_name']):
        return False

    return True
