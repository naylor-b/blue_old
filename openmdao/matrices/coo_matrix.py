"""Define the CooMatrix class."""
from __future__ import division

import numpy
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix
from six.moves import range
from six import iteritems

from openmdao.matrices.matrix import Matrix, _compute_index_map


class CooMatrix(Matrix):
    """Sparse matrix in Coordinate list format."""

    def _build_sparse(self, num_rows, num_cols):
        """Allocate the data, rows, and cols for the sparse matrix.

        Args
        ----
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            data, rows, cols that can be used to construct a sparse matrix.
        """
        counter = 0

        submat_meta_iter = ((self._out_submats, self._out_metadata),
                            (self._in_submats, self._in_metadata))

        for submats, metadata in submat_meta_iter:
            for key in submats:
                info = submats[key][0]
                value = info['value']
                ind1 = counter

                if value is not None:
                    jac = info['value']
                    if isinstance(jac, ndarray):
                        counter += jac.size
                    elif isinstance(jac, (coo_matrix, csr_matrix)):
                        counter += jac.data.size
                    elif isinstance(jac, list):
                        counter += len(jac[0])
                else:
                    if info['rows'] is None:  # dense subjac
                        counter += numpy.prod(info['shape'])
                    else:
                        counter += len(info['rows'])

                ind2 = counter
                metadata[key] = (ind1, ind2)

        data = numpy.empty(counter)
        rows = numpy.empty(counter, int)
        cols = numpy.empty(counter, int)

        for submats, metadata in submat_meta_iter:
            for key in submats:
                info, irow, icol, src_indices = submats[key]
                shape = info['shape']
                jac = info['value']
                dense = ((info['rows'] is None and jac is None) or 
                         isinstance(jac, ndarray))

                ind1, ind2 = metadata[key]
                idxs = None

                if dense:
                    rowrange = numpy.arange(shape[0], dtype=int)

                    if src_indices is None:
                        colrange = numpy.arange(shape[1], dtype=int)
                    else:
                        colrange = numpy.array(src_indices, dtype=int)

                    ncols = colrange.size
                    subrows = rows[ind1:ind2]
                    subcols = cols[ind1:ind2]

                    for i, row in enumerate(rowrange):
                        subrows[i * ncols: (i + 1) * ncols] = row
                        subcols[i * ncols: (i + 1) * ncols] = colrange

                    rows[ind1:ind2] += irow
                    cols[ind1:ind2] += icol

                    if jac is not None:
                        data[ind1:ind2] = jac.flat

                else:  #  sparse
                    if isinstance(jac, (coo_matrix, csr_matrix)):
                        jac = jac.tocoo()
                        jdata = jac.data
                        jrows = jac.row
                        jcols = jac.col
                    elif isinstance(jac, list):
                        jdata = jac[0]
                        jrows = jac[1]
                        jcols = jac[2]
                    else:  # value not provided
                        jdata = None
                        jrows = info['rows']
                        jcols = info['cols']

                    if src_indices is None:
                        if jdata is not None:
                            data[ind1:ind2] = jdata
                        rows[ind1:ind2] = jrows + irow
                        cols[ind1:ind2] = jcols + icol
                    else:
                        irows, icols, idxs = _compute_index_map(jrows, jcols,
                                                                irow, icol,
                                                                src_indices)
                        if jdata is not None:
                            data[ind1:ind2] = jdata[idxs]
                        rows[ind1:ind2] = irows
                        cols[ind1:ind2] = icols

                if metadata is self._in_metadata:
                    metadata[key] = (ind1, ind2, idxs)
                else:
                    metadata[key] = slice(ind1, ind2)

        return data, rows, cols

    def _build(self, num_rows, num_cols):
        """Allocate the matrix.

        Args
        ----
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        """
        data, rows, cols = self._build_sparse(num_rows, num_cols)

        for key in self._in_metadata:
            ind1, ind2, idxs = self._in_metadata[key]
            if idxs is None:
                self._in_metadata[key] = slice(ind1, ind2)
            else:
                # store reverse indices to avoid copying subjac data during
                # update_submat.
                self._in_metadata[key] = numpy.argsort(idxs) + ind1

        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols))

    def _update_submat(self, submats, metadata, key, jac):
        """Update the values of a sub-jacobian.

        Args
        ----
        submats : dict
            dictionary of sub-jacobian data keyed by (out_ind, in_ind).
        metadata : dict
            implementation-specific data for the sub-jacobians.
        key : (int, int)
            the global output and input variable indices.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        if isinstance(jac, ndarray):
            self._matrix.data[metadata[key]] = jac.flat
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix.data[metadata[key]] = jac.data
        elif isinstance(jac, list):
            self._matrix.data[metadata[key]] = jac[0]

    def _prod(self, in_vec, mode):
        """Perform a matrix vector product.

        Args
        ----
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        if mode == 'fwd':
            return self._matrix.dot(in_vec)
        elif mode == 'rev':
            return self._matrix.T.dot(in_vec)
