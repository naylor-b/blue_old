"""Define the base Jacobian class."""
from __future__ import division
import numpy
from scipy.sparse import coo_matrix, csr_matrix
from six.moves import range

from openmdao.utils.generalized_dict import OptionsDictionary


class Jacobian(object):
    """Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Attributes
    ----------
    _top_name : str
        name of the system at which we allocate the global Jacobian.
    _assembler : <Assembler>
        pointer to the assembler.
    _system : <System>
        pointer to the system that is currently operating on this Jacobian.
    _subjacs : dict
        dictionary containing the user-supplied external sub-Jacobians.
    _int_mtx : <Matrix>
        global internal Jacobian.
    _ext_mtx : <Matrix>
        global external Jacobian.
    _keymap : dict
        Mapping of original (output, input) key to (output, source) in cases
        where the input has src_indices.
    _iter_list : [(out_name, in_name), ...]
        list of output-input pairs to iterate over.
    options : <OptionsDictionary>
        options dictionary.
    """

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs : dict
            options dictionary.
        """
        self._top_name = None
        self._assembler = None
        self._system = None

        self._subjacs = {}
        self._int_mtx = None
        self._ext_mtx = None
        self._keymap = {}
        self._iter_list = []

        self.options = OptionsDictionary()
        self.options.update(kwargs)

    def _key2size(self, key):
        """Return the sizes of the variables making up the key tuple.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        out_size : int
            local size of the output variable.
        in_size : int
            local size of the input variable.
        """
        out_name, in_name = key
        if in_name in self._system._inputs:
            in_size = self._system._inputs._views_flat[in_name].size
        else:
            in_size = self._system._outputs[in_name].size

        return len(self._system._outputs._views_flat[out_name]), in_size

    def _key2idxs(self, key):
        """Map output-input pair name to corresponding indices.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        out_ind : int
            global index of output variable.
        in_ind : int
            global index of input variable.
        typ : str
            'input' or 'output'.
        """
        out_name, in_name = key

        in_ind = self._system._var_allprocs_indices['input'].get(in_name)
        if in_ind is not None:
            return (self._system._var_allprocs_indices['output'][out_name],
                    in_ind, 'input')

        return (self._system._var_allprocs_indices['output'][out_name],
                self._system._var_allprocs_indices['output'][in_name],
                'output')

    def _negate(self, key):
        """Multiply this sub-Jacobian by -1.0, for explicit variables.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        """
        ikey = self._key2idxs(key)
        jac = self._subjacs[ikey]

        if isinstance(jac, numpy.ndarray):
            self._subjacs[ikey] = -jac
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._subjacs[ikey].data *= -1.0  # DOK not supported
        elif len(jac) == 3:
            self._subjacs[ikey][0] *= -1.0
        elif len(jac) == 2:
            # In this case, negation is not necessary because sparse FD
            # works on the residuals which already contains the negation
            pass

    def _precompute_iter(self):
        """Assemble list of output-input pairs by name."""
        system = self._system

        self._iter_list = []
        for re_name in system._var_myproc_names['output']:
            re_ind = system._var_allprocs_indices['output'][re_name]

            for out_name in system._var_myproc_names['output']:
                out_ind = system._var_allprocs_indices['output'][out_name]

                if (re_ind, out_ind, 'output') in self._subjacs:
                    self._iter_list.append((re_name, out_name))

            for in_name in system._var_myproc_names['input']:
                in_ind = system._var_allprocs_indices['input'][in_name]

                if (re_ind, in_ind, 'input') in self._subjacs:
                    self._iter_list.append((re_name, in_name))

    def __contains__(self, key):
        """Map output-input pairs names to indices.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        boolean
            return whether sub-Jacobian has been defined.
        """
        return self._key2idxs(key) in self._subjacs

    def __iter__(self):
        """Return iterator from pre-computed _iter_list.

        Returns
        -------
        listiterator
            iterator returning (out_name, in_name) pairs.
        """
        return iter(self._iter_list)

    def __setitem__(self, key, jac):
        """Set sub-Jacobian.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        jac : int or float or ndarray or list[3] or tuple[3]
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        system = self._system
        out_ind, in_ind, typ = self._key2idxs(key)
        out_size, in_size = self._key2size(key)

        if numpy.isscalar(jac):
            jac = numpy.array([jac], float).reshape((out_size, in_size))
        elif isinstance(jac, (numpy.ndarray, coo_matrix, csr_matrix)):
            pass
        elif isinstance(jac, (tuple, list)):
            if len(jac) != 3:
                raise ValueError("Sub-jacobian of type '%s' for key %s has "
                                 "the wrong size (%d)." %
                                 (type(jac).__name__, key, len(jac)))
            if isinstance(jac, tuple):
                jac = list(jac)
        else:
            raise TypeError("Sub-jacobian of type '%s' for key %s is "
                            "not supported." % (type(jac).__name__, key))

        ind = system._var_myproc_names['output'].index(key[0])
        r_factor = system._scaling_to_norm['residual'][ind, 1]

        ind = system._var_myproc_names[typ].index(key[1])
        c_factor = system._scaling_to_norm[typ][ind, 1]

        if isinstance(jac, numpy.ndarray):
            jac *= r_factor / c_factor
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            jac.data *= r_factor / c_factor
        elif len(jac) == 3:
            jac[0] *= r_factor / c_factor

        self._subjacs[out_ind, in_ind, typ] = jac

    def __getitem__(self, key):
        """Get sub-Jacobian.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        jac : ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        system = self._system
        out_ind, in_ind, typ = self._key2idxs(key)
        jac = self._subjacs[out_ind, in_ind, typ]

        ind = system._var_myproc_names['output'].index(key[0])
        r_factor = system._scaling_to_phys['residual'][ind, 1]

        ind = system._var_myproc_names[typ].index(key[1])
        c_factor = system._scaling_to_phys[typ][ind, 1]

        if isinstance(jac, numpy.ndarray):
            jac *= r_factor / c_factor
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            jac.data *= r_factor / c_factor
        elif len(jac) == 3:
            jac[0] *= r_factor / c_factor

        return jac

    def _initialize(self):
        """Allocate the global matrices."""
        pass

    def _update(self):
        """Read the user's sub-Jacobians and set into the global matrix."""
        pass

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
        pass
