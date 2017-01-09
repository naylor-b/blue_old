"""Define the Component class."""

from __future__ import division

import collections

import numpy
from six import string_types

from openmdao.core.system import System


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

        # add this here even though we don't know the index yet, so we
        # can use the dict for fast containment checks later.
        self._var_allprocs_indices['input'][name] = None

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

        # add this here even though we don't know the index yet, so we
        # can use the dict for fast containment checks later.
        self._var_allprocs_indices['output'][name] = None

    def set_subjac_info(self, of, wrt, rows=None, cols=None,
                        approx=None, dependent=True, val=None):
        """Store subjacobian metadata for later use.

        Args
        ----
        of : str or list of str
            The name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or list of str
            The name of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.
        rows : ndarray of int or None
            Row indices for each nonzero entry.  For sparse subjacobians only.
        cols : ndarray of int or None
            Column indices for each nonzero entry.  For sparse subjacobians only.
        approx : str(None)
            Type of approximation ('fd' or 'cs') or None.
        dependent : bool(True)
            If False, specifies no dependence between the output(s) and the
            input(s).
        val : float or ndarray of float
            Value of subjacobian.

        """
        oflist = [of] if isinstance(of, string_types) else of
        wrtlist = [wrt] if isinstance(wrt, string_types) else wrt

        for of in oflist:
            for wrt in wrtlist:
                meta = {
                    'rows': rows,
                    'cols': cols,
                    'approx': approx,
                    'dependent': dependent,
                    'value': val,
                }

                # set shape metadata
                ofsize = numpy.prod(self._var2meta[of]['shape'])
                wrtsize = numpy.prod(self._var2meta[wrt]['shape'])
                meta['shape'] = (ofsize, wrtsize)
                self._subjacs_info.append((of, wrt, meta))

    def _setup_vector(self, vectors, vector_var_ids, use_ref_vector):
        r"""Add this vector and assign sub_vectors to subsystems.

        Sets the following attributes:

        - _vectors
        - _vector_transfers
        - _inputs*
        - _outputs*
        - _residuals*
        - _transfers*

        \* If vec_name is None - i.e., we are setting up the nonlinear vector

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
        if vectors['input']._name is None:
            names = self._var_myproc_names['input']
            inputs = self._inputs
            for i, meta in enumerate(self._var_myproc_metadata['input']):
                inputs[names[i]] = meta['value']

        if vectors['output']._name is None:
            names = self._var_myproc_names['output']
            outputs = self._outputs
            for i, meta in enumerate(self._var_myproc_metadata['output']):
                outputs[names[i]] = meta['value']
