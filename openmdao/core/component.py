"""Define the Component class."""

from __future__ import division

from fnmatch import fnmatchcase
from six import string_types
import numpy

from openmdao.core.system import System, PathData


class Component(System):
    """Base Component class; not to be directly instantiated.

    Attributes
    ----------
    _var2meta : dict
        A mapping of local variable name to its metadata.
    """

    INPUT_DEFAULTS = {
        'shape': (1,),
        'units': '',
        'var_set': 0,
        'indices': None,
    }

    OUTPUT_DEFAULTS = {
        'shape': (1,),
        'units': '',
        'var_set': 0,
        'lower': None,
        'upper': None,
        'ref': 1.0,
        'ref0': 0.0,
        'res_units': '',
        'res_ref': 1.0,
        'res_ref0': 0.0,
    }

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs: dict of keyword arguments
            available here and in all descendants of this system.
        """
        super(Component, self).__init__(**kwargs)
        self._var2meta = {}

    def add_input(self, name, val=1.0, **kwargs):
        """Add an input variable to the component.

        Args
        ----
        name : str60
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        metadata = self.INPUT_DEFAULTS.copy()
        metadata.update(kwargs)

        metadata['value'] = val
        if 'indices' in kwargs:
            metadata['indices'] = numpy.array(kwargs['indices'])
            metadata['shape'] = metadata['indices'].shape
        elif 'shape' not in kwargs and isinstance(val, numpy.ndarray):
            metadata['shape'] = val.shape

        self._var_allprocs_names['input'].append(name)
        self._var_myproc_names['input'].append(name)
        self._var_myproc_metadata['input'].append(metadata)
        self._var2meta[name] = metadata

    def add_output(self, name, val=1.0, **kwargs):
        """Add an output variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        metadata = self.OUTPUT_DEFAULTS.copy()
        metadata.update(kwargs)

        metadata['value'] = val
        if 'shape' not in kwargs and isinstance(val, numpy.ndarray):
            metadata['shape'] = val.shape

        self._var_allprocs_names['output'].append(name)
        self._var_myproc_names['output'].append(name)
        self._var_myproc_metadata['output'].append(metadata)
        self._var2meta[name] = metadata

    def declare_partial_derivs(self, of, wrt, dependent=True,
                               rows=None, cols=None, val=None):
        """Store subjacobian metadata for later use.

        Args
        ----
        of : str or list of str
            The name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or list of str
            The name of the variables that derivatives are taken with respect
            to. This can contain the name of any input or output variable.
            May also contain a glob pattern.
        dependent : bool(True)
            If False, specifies no dependence between the output(s) and the
            input(s). This is only necessary in the case of a sparse global
            jacobian, because if 'dependent=False' is not specified and
            declare_partial_derivs is not called for a given pair, then a dense
            matrix of zeros will be allocated in the sparse global jacobian
            for that pair.  In the case of a dense global jacobian it doesn't
            matter because the space for a dense subjac will always be
            allocated for every pair.
        rows : ndarray of int or None
            Row indices ffor nonzero entries.  For sparse subjacobians only.
        cols : ndarray of int or None
            Column indices for nonzero entries. For sparse subjacobians only.
        val : float or ndarray of float
            Value of subjacobian.  If rows and cols are not None, this will
            contain the values found at each (row, col) location in the subjac.

        """
        oflist = [of] if isinstance(of, string_types) else of
        wrtlist = [wrt] if isinstance(wrt, string_types) else wrt

        if isinstance(rows, (list, tuple)):
            rows = numpy.array(rows, dtype=int)
        if isinstance(cols, (list, tuple)):
            cols = numpy.array(cols, dtype=int)

        for of in oflist:
            for wrt in wrtlist:
                meta = {
                    'rows': rows,
                    'cols': cols,
                    'value': val,
                    'dependent': dependent,
                }
                self._subjacs_info.append((of, wrt, meta))

    def _set_partial_derivs_meta(self):
        """Set subjacobian metadata into our jacobian."""
        indices = self._var_allprocs_indices
        oldsys = self._jacobian._system
        self._jacobian._system = self

        outs = self._var_allprocs_names['output']
        ins = self._var_allprocs_names['input']

        for of, wrt, meta in self._subjacs_info:
            ofmatches = [n for n in outs if n == of or fnmatchcase(n, of)]
            for typ, vnames in (('output', outs), ('input', ins)):
                for wrtname in vnames:
                    if wrtname == wrt or fnmatchcase(wrtname, wrt):
                        for ofmatch in ofmatches:
                            key = (ofmatch, wrtname)
                            self._jacobian._set_subjac_meta(key, meta)

        self._jacobian._system = oldsys

    def _setup_variables(self, recurse=False):
        """Assemble variable metadata and names lists.

        Sets the following attributes:
            _var_allprocs_names
            _var_myproc_names
            _var_myproc_metadata

        Args
        ----
        recurse : boolean
            Ignored.
        """
        super(Component, self)._setup_variables(False)

        self._var_pathdict = {}
        self._var_name2path = {}
        for typ in ['input', 'output']:
            self._var_allprocs_pathnames[typ] = [
                '.'.join((self.pathname, n)) for n in
                    self._var_allprocs_names[typ]
            ]
            for idx, name in enumerate(self._var_allprocs_names[typ]):
                path = self._var_allprocs_pathnames[typ][idx]
                self._var_pathdict[path] = PathData(name, idx, typ)
                self._var_name2path[name] = (path,)

    def _setup_vector(self, vectors, vector_var_ids, use_ref_vector):
        r"""Add this vector and assign sub_vectors to subsystems.

        Sets the following attributes:

        - _vectors
        - _vector_transfers
        - _inputs*
        - _outputs*
        - _residuals*
        - _transfers*

        \* If vec_name is 'nonlinear'

        Args
        ----
        vectors : {'input': <Vector>, 'output': <Vector>, 'residual': <Vector>}
            <Vector> objects corresponding to 'name'.
        vector_var_ids : ndarray[:]
            integer array of all relevant variables for this vector.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        super(Component, self)._setup_vector(vectors, vector_var_ids,
                                             use_ref_vector)

        # Components must load their initial input and output values into the
        # vectors.

        # Note: It's possible for meta['value'] to not match
        #       meta['shape'], and input and output vectors are sized according
        #       to shape, so if, for example, value is not specified it
        #       defaults to 1.0 and the shape can be anything, resulting in the
        #       value of 1.0 being broadcast into all values in the vector
        #       that were allocated according to the shape.
        if vectors['input']._name is 'nonlinear':
            names = self._var_myproc_names['input']
            inputs = self._inputs
            for i, meta in enumerate(self._var_myproc_metadata['input']):
                inputs[names[i]] = meta['value']

        if vectors['output']._name is 'nonlinear':
            names = self._var_myproc_names['output']
            outputs = self._outputs
            for i, meta in enumerate(self._var_myproc_metadata['output']):
                outputs[names[i]] = meta['value']
