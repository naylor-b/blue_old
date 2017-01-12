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

        for key, (jac, irow, icol, src_indices) in iteritems(self._submats):
            ind1 = counter
            if isinstance(jac, ndarray):
                counter += jac.size
            elif isinstance(jac, (coo_matrix, csr_matrix)):
                counter += jac.data.size
            elif isinstance(jac, list):
                counter += len(jac[0])
            ind2 = counter
            self._metadata[key] = (ind1, ind2)

        data = numpy.empty(counter)
        rows = numpy.empty(counter, int)
        cols = numpy.empty(counter, int)

        for key, (jac, irow, icol, src_indices) in iteritems(self._submats):
            ind1, ind2 = self._metadata[key]
            idxs = None

            if isinstance(jac, ndarray):
                rowrange = numpy.arange(jac.shape[0], dtype=int)

                if src_indices is None:
                    colrange = numpy.arange(jac.shape[1], dtype=int)
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
                data[ind1:ind2] = jac.flat

            elif isinstance(jac, (coo_matrix, csr_matrix)):
                coojac = jac.tocoo()
                if src_indices is None:
                    data[ind1:ind2] = coojac.data
                    rows[ind1:ind2] = coojac.row + irow
                    cols[ind1:ind2] = coojac.col + icol
                else:
                    irows, icols, idxs = _compute_index_map(coojac.row,
                                                            coojac.col,
                                                            irow, icol,
                                                            src_indices)
                    data[ind1:ind2] = jac.data[idxs]
                    rows[ind1:ind2] = irows
                    cols[ind1:ind2] = icols

            elif isinstance(jac, list):
                if src_indices is None:
                    data[ind1:ind2] = jac[0]
                    rows[ind1:ind2] = irow + jac[1]
                    cols[ind1:ind2] = icol + jac[2]
                else:
                    irows, icols, idxs = _compute_index_map(jac[1],
                                                            jac[2],
                                                            irow, icol,
                                                            src_indices)
                    data[ind1:ind2] = jac[0][idxs]
                    rows[ind1:ind2] = irows
                    cols[ind1:ind2] = icols

            self._metadata[key] = (ind1, ind2, idxs)

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

        metadata = self._metadata
        for key, (ind1, ind2, idxs) in iteritems(metadata):
            if idxs is None:
                metadata[key] = slice(ind1, ind2)
            else:
                # store reverse indices to avoid copying subjac data during
                # update_submat.
                metadata[key] = numpy.argsort(idxs) + ind1

        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols))

    def _update_submat(self, key, jac):
        """Update the values of a sub-jacobian.

        Args
        ----
        key : (int, int)
            the global output and input variable indices.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        if isinstance(jac, ndarray):
            self._matrix.data[self._metadata[key]] = jac.flat
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix.data[self._metadata[key]] = jac.data
        elif isinstance(jac, list):
            self._matrix.data[self._metadata[key]] = jac[0]

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
