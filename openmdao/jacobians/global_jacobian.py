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
        meta_in = self._system._var_myproc_metadata['input']
        meta_out = self._system._var_myproc_metadata['output']
        out_names = self._system._var_allprocs_names['output']
        ivar1, ivar2 = self._system._var_allprocs_range['output']

        self._int_mtx = self.options['matrix_class'](self._system.comm)
        self._ext_mtx = self.options['matrix_class'](self._system.comm)

        out_offsets = {i: self._get_var_range(i, 'output')[0]
                       for i in var_indices['output']}
        in_offsets = {i: self._get_var_range(i, 'input')[0]
                      for i in indices['input']}
        src_indices = {i: meta[j]['indices']
                       for j, i in enumerate(indices['input'])}

        for re_var_all in indices['output']:
            re_offset = out_offsets[re_var_all]

            for out_var_all in indices['output']:
                key = (re_var_all, out_var_all, 'output')
                if key in self._subjacs:
                    jac = self._subjacs[key]

                    self._int_mtx._add_submat(
                        key, jac, re_offset, out_offsets[out_var_all], None)

            for in_var_all in indices['input']:
                key = (re_var_all, in_var_all, 'input')
                if key in self._subjacs:
                    jac = self._subjacs[key]

                    out_var_all = self._assembler._input_src_ids[in_var_all]
                    if ivar1 <= out_var_all < ivar2:
                        if src_indices[in_var_all] is None:
                            self._keymap[key] = key
                            self._int_mtx._add_submat(
                                key, jac, re_offset, out_offsets[out_var_all],
                                None)
                        else:
                            # need to add an entry for d(output)/d(source)
                            # instead of d(output)/d(input) when we have
                            # src_indices
                            key2 = (key[0],
                                    self._assembler._input_src_ids[in_var_all],
                                    'output')
                            self._keymap[key] = key2
                            self._int_mtx._add_submat(
                                key2, jac, re_offset, out_offsets[out_var_all],
                                src_indices[in_var_all])
                    else:
                        self._ext_mtx._add_submat(
                            key, jac, re_offset, in_offsets[in_var_all], None)

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

        for re_var_all in indices['output']:
            for out_var_all in indices['output']:
                key = (re_var_all, out_var_all, 'output')
                if key in self._subjacs:
                    self._int_mtx._update_submat(key, self._subjacs[key])

            for in_var_all in indices['input']:
                key = (re_var_all, in_var_all, 'input')
                if key in self._subjacs:
                    out_var_all = self._assembler._input_src_ids[in_var_all]
                    if ivar1 <= out_var_all < ivar2:
                        self._int_mtx._update_submat(self._keymap[key],
                                                     self._subjacs[key])
                    else:
                        self._ext_mtx._update_submat(key,
                                                     self._subjacs[key])

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

    def _set_subjac_info(self, key, meta):
        """Store subjacobian metadata.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        meta : dict
            Metadata dictionary for the subjacobian.
        """
        out_ind, in_ind, typ = self._key2idxs(key)
        out_size, in_size = self._key2size(key)

        if typ == 'input':
            self._subjacs_in_info[(out_ind, in_ind)] = (meta, (out_size, in_size))
        else:
            self._subjacs_out_info[(out_ind, in_ind)] = (meta, (out_size, in_size))

        val = meta['value']
        if val is not None:
            if meta['rows'] is not None:
                val = [val, meta['rows'], meta['cols']]
            self.__setitem__(key, val)
