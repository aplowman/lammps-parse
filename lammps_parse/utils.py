"""`lammps_parse.utils.py`"""

import os
import re

import numpy as np


def format_args_check(**kwargs):
    'Check types of parameters used in the `format_arr` function.'

    if 'depth' in kwargs and not isinstance(kwargs['depth'], int):
        raise ValueError('`depth` must be an integer.')

    if 'indent' in kwargs and not isinstance(kwargs['indent'], str):
        raise ValueError('`indent` must be a string.')

    if 'col_delim' in kwargs and not isinstance(kwargs['col_delim'], str):
        raise ValueError('`col_delim` must be a string.')

    if 'row_delim' in kwargs and not isinstance(kwargs['row_delim'], str):
        raise ValueError('`row_delim` must be a string.')

    if 'dim_delim' in kwargs and not isinstance(kwargs['dim_delim'], str):
        raise ValueError('`dim_delim` must be a string.')

    if 'format_spec' in kwargs and not isinstance(kwargs['format_spec'],
                                                  (str, list)):
        raise ValueError('`format_spec` must be a string or list of strings.')

    if 'assign' in kwargs:
        if not isinstance(kwargs['assign'], str):
            raise ValueError('`assign` must be a string.')


def format_arr(arr, depth=0, indent='\t', col_delim='\t', row_delim='\n', dim_delim='\n',
               format_spec='{}'):
    """Get a string representation of a Numpy array, formatted with indents.

    Parameters
    ----------
    arr : ndarray or list of ndarray
        Array of any shape to format as a string, or list of arrays whose shapes match
        except for the final dimension, in which case the arrays will be formatted
        horizontally next to each other.
    depth : int, optional
        The indent depth at which to begin the formatting.
    indent : str, optional
        The string used as the indent. The string which indents each line of the array is
        equal to (`indent` * `depth`).
    col_delim : str, optional
        String to delimit columns (the innermost dimension of the array). Default is tab
        character, \t.
    row_delim : str, optional
        String to delimit rows (the second-innermost dimension of the array). Default is
        newline character, \n.
    dim_delim : str, optional
        String to delimit outer dimensions. Default is newline character, \n.
    format_spec : str or list of str, optional
        Format specifier for the array or a list of format specifiers, one for each array
        listed in `arr`.

    Returns
    -------
    str
        Arrays formatted as a string.

    """

    # Validation:
    format_args_check(depth=depth, indent=indent, col_delim=col_delim,
                      row_delim=row_delim, dim_delim=dim_delim,
                      format_spec=format_spec)

    if isinstance(arr, np.ndarray):
        arr = [arr]

    out_shape = list(set([i.shape[:-1] for i in arr]))

    if len(out_shape) > 1:
        msg = 'Array shapes must be identical apart from the innermost dimension.'
        raise ValueError(msg)

    if not isinstance(arr, (list, np.ndarray)):
        msg = ('Cannot format as array, object is not an array or list of arrays: '
               'type is {}')
        raise ValueError(msg.format(type(arr)))

    if isinstance(format_spec, str):
        format_spec = [format_spec] * len(arr)

    elif isinstance(format_spec, list):

        fs_err_msg = ('`format_spec` must be a string or list of N strings where N is the'
                      ' number of arrays specified in `arr`.')
        if not all([isinstance(i, str)
                    for i in format_spec]) or len(format_spec) != len(arr):
            raise ValueError(fs_err_msg)

    arr_list = arr
    out = ''
    dim_seps = ''
    d = arr_list[0].ndim

    if d == 1:
        out += (indent * depth)

        for sa_idx, sub_arr in enumerate(arr_list):
            for col_idx, col in enumerate(sub_arr):
                out += format_spec[sa_idx].format(col)
                if (col_idx < len(sub_arr) - 1):
                    out += col_delim

        out += row_delim

    else:

        if d > 2:
            dim_seps = dim_delim * (d - 2)

        sub_arr = []
        for i in range(out_shape[0][0]):

            sub_arr_lst = []
            for j in arr_list:
                sub_arr_lst.append(j[i])

            sub_arr.append(format_arr(sub_arr_lst, depth, indent, col_delim,
                                      row_delim, dim_delim, format_spec))

        out = dim_seps.join(sub_arr)

    return out
