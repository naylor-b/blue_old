"""Define the GlobalJacobian class."""
from __future__ import division

import numpy
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.utils.generalized_dict import OptionsDictionary


class GlobalJacobian(Jacobian):
    """Assemble dense global <Jacobian>.

    Attributes
    ----------
    _subjacs_in_info : dict
        Dict of subjacobian metadata keyed on (resid_idx, input_idx).
    _subjacs_out_info : dict
        Dict of subjacobian metadata keyed on (resid_idx, output_idx).
    """

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs : dict
            options dictionary.
        """
        super(GlobalJacobian, self).__init__()
        self.options.declare('matrix_class', value=DenseMatrix,
                             desc='<Matrix> class to use in this <Jacobian>.')
        self.options.update(kwargs)

        # dicts of subjacobian metadata keyed by (resid_index, in_index)
        # and (resid_index, out_index) respectively
        self._subjacs_in_info = {}
        self._subjacs_out_info = {}

    def _get_var_range(self, ivar_all, typ):
        """Look up the variable name and <Jacobian> index range.

        Args
        ----
        ivar_all : int
            index of a variable in the global ordering.
        typ : str
            'input' or 'output'.

        Returns
        -------
        int
            the starting index in the Jacobian.
        int
            the ending index in the Jacobian.
        """
        sizes_all = self._assembler._variable_sizes_all
        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]
        ivar_all0 = self._system._var_allprocs_range['output'][0]

        ind1 = numpy.sum(sizes_all['output'][iproc, ivar_all0:ivar_all])
        ind2 = numpy.sum(sizes_all['output'][iproc, ivar_all0:ivar_all + 1])

        return ind1, ind2

    def _initialize(self):
        """Allocate the global matrices."""
        # var_indices are the *global* indices for variables on this proc
        var_indices = self._system._var_myproc_indices
        meta = self._system._var_myproc_metadata['input']
        ivar1, ivar2 = self._system._var_allprocs_range['output']

        self._int_mtx = self.options['matrix_class'](self._system.comm)
        self._ext_mtx = self.options['matrix_class'](self._system.comm)

        out_offsets = {i: self._get_var_range(i, 'output')[0]
                       for i in var_indices['output']}
        in_offsets = {i: self._get_var_range(i, 'input')[0]
                      for i in var_indices['input']}
        src_indices = {i: meta[j]['indices']
                       for j, i in enumerate(var_indices['input'])}

        for re_idx_all in var_indices['output']:
            re_offset = out_offsets[re_idx_all]

            for out_idx_all in var_indices['output']:
                key = (re_idx_all, out_idx_all)
                if key in self._subjacs_out_info:
                    info = self._subjacs_out_info[key]

                    self._int_mtx._out_submats[key] = (info, re_offset,
                                                       out_offsets[out_idx_all],
                                                       None)

            for in_idx_all in var_indices['input']:
                key = (re_idx_all, in_idx_all)
                self._keymap[key] = key

                if key in self._subjacs_in_info:
                    info = self._subjacs_in_info[key]

                    out_idx_all = self._assembler._input_src_ids[in_idx_all]
                    if ivar1 <= out_idx_all < ivar2:
                        if src_indices[in_idx_all] is None:
                            self._int_mtx._in_submats[key] = (info, re_offset,
                                                              out_offsets[out_idx_all],
                                                              None)
                        else:
                            # need to add an entry for d(output)/d(source)
                            # instead of d(output)/d(input) when we have
                            # src_indices
                            key2 = (key[0], out_idx_all)
                            self._keymap[key] = key2
                            self._int_mtx._out_submats[key2] = (info, re_offset,
                                                                out_offsets[out_idx_all],
                                                                src_indices[in_idx_all])
                    else:
                        self._ext_mtx._in_submats[key] = (info, re_offset,
                                                          in_offsets[in_idx_all],
                                                          None)

        out_size = numpy.sum(
            self._assembler._variable_sizes_all['output'][ivar1:ivar2])

        ind1, ind2 = self._system._var_allprocs_range['input']
        in_size = numpy.sum(
            self._assembler._variable_sizes_all['input'][ind1:ind2])

        self._int_mtx._build(out_size, out_size)
        self._ext_mtx._build(out_size, in_size)

    def _update(self):
        """Read the user's sub-Jacobians and set into the global matrix."""
        # var_var_indices are the *global* indices for variables on this proc
        var_indices = self._system._var_myproc_indices
        ivar1, ivar2 = self._system._var_allprocs_range['output']

        for re_idx_all in var_indices['output']:
            for out_idx_all in var_indices['output']:
                key = (re_idx_all, out_idx_all)
                if key in self._out_dict:
                    self._int_mtx._out_update_submat(key, self._out_dict[key])

            for in_idx_all in var_indices['input']:
                key = (re_idx_all, in_idx_all)
                if key in self._in_dict:
                    out_idx_all = self._assembler._input_src_ids[in_idx_all]
                    if ivar1 <= out_idx_all < ivar2:
                        self._int_mtx._in_update_submat(self._keymap[key],
                                                        self._in_dict[key])
                    else:
                        self._ext_mtx._in_update_submat(key,
                                                        self._in_dict[key])

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """Compute matrix-vector product.

        Args
        ----
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        int_mtx = self._int_mtx
        ext_mtx = self._ext_mtx

        if mode == 'fwd':
            d_residuals.iadd_data(int_mtx._prod(d_outputs.get_data(), mode))
            d_residuals.iadd_data(ext_mtx._prod(d_inputs.get_data(), mode))
        elif mode == 'rev':
            d_outputs.iadd_data(int_mtx._prod(d_residuals.get_data(), mode))
            d_inputs.iadd_data(ext_mtx._prod(d_residuals.get_data(), mode))

    def _set_subjac_info(self, strkey, idxkey, meta, typ):
        """Store subjacobian metadata.

        Args
        ----
        strkey : (ofname, wrtname)
            A pair of names used as key for __setitem__.
        idxkey : (of, wrt)
            A pair of indices used to key into the info dicts.
        meta : dict
            Metadata dictionary for the subjacobian.
        typ : str
            Indictates the I/O type of the wrt variable.
        """
        if typ == 'input':
            self._subjacs_in_info[idxkey] = meta
        else:
            self._subjacs_out_info[idxkey] = meta

        val = meta['value']
        if val is not None:
            if meta['rows'] is not None:
                val = [val, meta['rows'], meta['cols']]
            self.__setitem__(strkey, val)
