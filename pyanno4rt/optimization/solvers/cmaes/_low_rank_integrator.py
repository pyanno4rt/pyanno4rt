"""Dynamical low-rank integrator."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import (
    copyto, diag, eye, float64, hstack, matmul, maximum, sqrt, zeros)
from numpy import sum as nsum
from numpy.linalg import norm, svd
from scipy.linalg import qr

# %% Dynamical low-rank integrator class


class LowRankIntegrator:
    """
    Dynamical low-rank integrator class.

    This class implements update steps for different types of dynamical \
    low-rank integrators.

    Parameters
    ----------
    name : {'fixedBUG', 'fixedsymmetricBUG', 'fixedaugBUG', 'augBUG', \
            'symmetricaugBUG', 'parBUG'}
        Name of the low-rank integrator.

    rank : int
        Initial rank of the approximation.

    truncation_tolerance_rel : float
        Relative tolerance of the rank truncation.

    truncation_tolerance_abs : float
        Absolute tolerance of the rank truncation.

    N_conserved_basis : int
        ...

    K_step : Callable
        ...

    L_step : Callable
        ...

    S_step : Callable
        ...
    """

    def __init__(
        self,
        name,
        rank,
        truncation_tolerance_rel,
        truncation_tolerance_abs,
        N_conserved_basis,
        K_step,
        L_step,
        S_step):

        # Get the input attributes
        self.name = name
        self.rank = rank
        self.truncation_tolerance_rel = truncation_tolerance_rel
        self.truncation_tolerance_abs = truncation_tolerance_abs
        self.N_conserved_basis = N_conserved_basis
        self.K_step = K_step
        self.L_step = L_step
        self.S_step = S_step

        # Get the update function
        updates = {
            'fixedBUG': self.fixedBUG_step,
            'fixedsymmetricBUG': self.fixedsymmetricBUG_step,
            'fixedaugBUG': self.fixedaugBUG_step,
            'augBUG': self.augBUG_step,
            'symmetricaugBUG': self.symmetricaugBUG_step,
            'parBUG': self.parBUG_step
            }
        self.update_func = updates.get(name, self.fixedsymmetricBUG_step)

        # Initialize buffers for the low-rank factors
        self._K = None
        self._Uhat = None
        self._L = None
        self._Vhat = None
        self._M = None
        self._N = None

        # Initialize the rank capacity
        self._capacity = None

        # Initialize the rank history
        self.rank_history = [rank]

    def set_buffers(
            self,
            number_of_variables,
            initial_rank,
            capacity=None):
        """
        Initialize the low-rank factor buffers.

        Parameters
        ----------
        number_of_variables : int
            Dimension of the search space (number of decision variables).

        capacity : int, default=None
            The desired buffer capacity.
        """

        # Check if a desired capacity has been passed
        if capacity is not None:

            # Clip the capacity by the number of variables
            self._capacity = min(2*number_of_variables, capacity)

        else:

            # Fall back to a default capacity
            self._capacity = min(
                2*number_of_variables, max(2*number_of_variables // 10, 100))

        #
        rank = min(initial_rank, self._capacity // 2)

        # Initialize K
        self._K = zeros(
            (number_of_variables, self._capacity), order='F', dtype=float64)
        self._K[:rank, :rank] = eye(rank, dtype=float64)

        # Initialize Uhat
        self._Uhat = zeros(
            (number_of_variables, self._capacity), order='F', dtype=float64)

        # Check if a symmetric integrator is used
        if 'symmetric' not in self.name.lower():

            # Initialize L
            self._L = zeros(
                (number_of_variables, self._capacity),
                dtype=float64, order='F')
            self._L[:rank, :rank] = eye(rank, dtype=float64)

            # Initialize Vhat
            self._Vhat = zeros(
                (number_of_variables, self._capacity),
                dtype=float64, order='F')

            # Initialize N
            self._N = zeros((self._capacity, self._capacity), dtype=float64)
            self._N[:rank, :rank] = eye(rank, dtype=float64)

        # Initialize M and N
        self._M = zeros((self._capacity, self._capacity), dtype=float64)
        self._M[:rank, :rank] = eye(rank, dtype=float64)

    def update(
            self,
            U,
            S,
            V,
            dt):
        """
        Update the low-rank factors.

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        # Get the updated factors
        U_new, S_new, V_new = self.update_func(U, S, V, dt)

        return U_new, S_new, V_new

    def fixedBUG_step(
            self,
            U,
            S,
            V,
            dt):
        """
        Perform a single step of the fixed-rank BUG integrator.

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        #
        matmul(U, S, out=self._K)
        matmul(V, S.T, out=self._L)

        #
        K_updated = self.K_step(self._K, V, dt)
        L_updated = self.L_step(U, self._L, dt)

        #
        Uhat, _ = qr(K_updated, mode='economic', check_finite=False)
        copyto(self._Uhat, Uhat)

        #
        Vhat, _ = qr(L_updated, mode='economic', check_finite=False)
        copyto(self._Vhat, Vhat)

        #
        matmul(self._Uhat.T, U, out=self._M)
        matmul(self._Vhat.T, V, out=self._N)

        #
        ext_S = self._M @ S @ self._N.T
        Shat = self.S_step(
            self._Uhat, ext_S, self._Vhat, self._Uhat, None, None, dt)

        #
        Shat += Shat.T
        Shat *= 0.5

        return self._Uhat, Shat, self._Vhat

    def fixedsymmetricBUG_step(
            self,
            U,
            S,
            _,
            dt):
        """
        Perform a single step of the fixed-rank symmetric BUG integrator.

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        #
        rank = U.shape[1]

        #
        K_slice = self._K[:, :rank]
        Uhat_slice = self._Uhat[:, :rank]

        #
        self._K[:, :rank] = U @ S[:rank, :rank]

        #
        K_updated = self.K_step(K_slice, U, dt)

        #
        Uhat, _ = qr(K_updated, mode='economic', check_finite=False)
        copyto(Uhat_slice, Uhat)

        #
        M_proj = self._M[:rank, :rank]
        matmul(Uhat_slice.T, U, out=M_proj)

        #
        ext_S = M_proj @ S[:rank, :rank] @ M_proj.T
        Shat = self.S_step(
            Uhat_slice, ext_S, Uhat_slice, Uhat_slice, None, None, dt)

        #
        Shat += Shat.T
        Shat *= 0.5

        return Uhat_slice, Shat, Uhat_slice

    def fixedaugBUG_step(
            self,
            U,
            S,
            V,
            dt):
        """
        Perform a single step of the fixed-rank augmented BUG integrator.

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        #
        rank = U.shape[1]
        max_rank = 2*rank

        #
        K_slice = self._K[:, :rank]
        K_aug = self._K[:, :max_rank]
        L_slice = self._L[:, :rank]
        L_aug = self._L[:, :max_rank]
        Uhat_aug = self._Uhat[:, :max_rank]
        Vhat_aug = self._Vhat[:, :max_rank]

        #
        matmul(U, S, out=K_slice)
        matmul(V, S.T, out=L_slice)

        #
        self.K_step(K_slice, V, dt)
        copyto(self._K[:, rank:max_rank], U)

        #
        self.L_step(L_slice, U, dt)
        copyto(self._L[:, rank:max_rank], V)

        #
        Uhat, _ = qr(K_aug, mode='economic', check_finite=False)
        copyto(self._Uhat[:, :Uhat.shape[1]], Uhat)

        #
        Vhat, _ = qr(L_aug, mode='economic', check_finite=False)
        copyto(self._Vhat[:, :Vhat.shape[1]], Vhat)

        #
        M_proj = self._M[:max_rank, :rank]
        N_proj = self._N[:max_rank, :rank]
        matmul(Uhat_aug.T, U, out=M_proj)
        matmul(Vhat_aug.T, V, out=N_proj)

        #
        ext_S = M_proj @ S @ N_proj.T
        Shat = self.S_step(
            Uhat_aug, ext_S, Vhat_aug, Uhat_aug, ext_S, Vhat_aug, dt)

        #
        Shat += Shat.T
        Shat *= 0.5

        return self.truncate(Uhat_aug, Shat, Vhat_aug, fixed=True)

    def augBUG_step(
            self,
            U,
            S,
            V,
            dt):
        """
        Perform a single step of the augmented BUG integrator.

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        #
        rank = U.shape[1]
        max_rank = 2*rank

        #
        K_slice = self._K[:, :rank]
        K_aug = self._K[:, :max_rank]
        L_slice = self._L[:, :rank]
        L_aug = self._L[:, :max_rank]
        Uhat_aug = self._Uhat[:, :max_rank]
        Vhat_aug = self._Vhat[:, :max_rank]

        #
        matmul(U, S, out=K_slice)
        matmul(V, S.T, out=L_slice)

        #
        self.K_step(K_slice, V, dt)
        copyto(self._K[:, rank:max_rank], U)

        #
        self.L_step(L_slice, U, dt)
        copyto(self._L[:, rank:max_rank], V)

        #
        Uhat, _ = qr(K_aug, mode='economic', check_finite=False)
        copyto(self._Uhat[:, :Uhat.shape[1]], Uhat)

        #
        Vhat, _ = qr(L_aug, mode='economic', check_finite=False)
        copyto(self._Vhat[:, :Vhat.shape[1]], Vhat)

        #
        M_proj = self._M[:max_rank, :rank]
        N_proj = self._N[:max_rank, :rank]
        matmul(Uhat_aug.T, U, out=M_proj)
        matmul(Vhat_aug.T, V, out=N_proj)

        #
        ext_S = M_proj @ S @ N_proj.T
        Shat = self.S_step(
            Uhat_aug, ext_S, Vhat_aug, Uhat_aug, ext_S, Vhat_aug, dt)

        #
        Shat += Shat.T
        Shat *= 0.5

        return self.truncate(Uhat_aug, Shat, Vhat_aug)

    def symmetricaugBUG_step(
            self,
            U,
            S,
            V,
            dt):
        """
        Perform a single step of the symmetric augmented BUG integrator.

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        #
        rank = U.shape[1]

        #
        rank_limit = self._capacity // 2

        #
        if rank > rank_limit:

            #
            max_rank = self._capacity

            #
            num_to_augment = max_rank - rank

            #
            U_to_copy = U[:, :num_to_augment]

        else:

            #
            max_rank = 2*rank

            #
            num_to_augment = rank

            #
            U_to_copy = U

        #
        K_slice = self._K[:, :rank]
        K_aug = self._K[:, :max_rank]
        Uhat_aug = self._Uhat[:, :max_rank]

        #
        self._K[:, :rank] = U @ S[:rank, :rank]

        #
        self.K_step(K_slice, V, dt)

        #
        if num_to_augment > 0:

            #
            copyto(self._K[:, rank:max_rank], U_to_copy)

        #
        Uhat, _ = qr(K_aug, mode='economic', check_finite=False)
        copyto(self._Uhat[:, :Uhat.shape[1]], Uhat)

        #
        M_proj = self._M[:max_rank, :rank]
        matmul(Uhat_aug.T, U, out=M_proj)

        #
        ext_S = M_proj @ S[:rank, :rank] @ M_proj.T
        Shat = self.S_step(
            Uhat_aug, ext_S, Uhat_aug, Uhat_aug, ext_S, Uhat_aug, dt)

        #
        Shat += Shat.T
        Shat *= 0.5

        return self.truncate(Uhat_aug, Shat, Uhat_aug)

    def parBUG_step(
            self,
            U,
            S,
            V,
            dt):
        """
        .

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        K = self.K_step(U @ S, V, dt)
        Utmp,_ = qr(hstack((U,K)))
        Utilde = Utmp[:,self.rank+1:]
        Uhat = hstack(U, Utilde)

        L = self.L_step(V @ S.T, U, dt)
        Vtmp,_ = qr(hstack((V,L)))
        Vtilde = Vtmp[:,self.rank+1:]
        Vhat = hstack(V, Vtilde)

        S = self.S_step(U, S, V, U, S, V, dt)

        Shat = zeros(2*self.rank,2*self.rank)
        Shat[:self.rank,:self.rank] = S
        Shat[:self.rank,:self.rank+1:] = L.T @ Vtilde
        Shat[self.rank+1:,1:self.rank] = Utilde.T @ K

        U, S, V = self.truncate(Uhat, Shat, Vhat)

        return U, S, V

    def truncate(
            self,
            U,
            S,
            V,
            fixed=False):
        """
        .

        Parameters
        ----------
        ...

        Returns
        -------
        ...
        """

        # Compute singular values of S and decide how to truncate:
        m = U.shape[0]
        n = V.shape[0]
        rMaxTotal = min(m, n)
        rMinTotal = 2

        if fixed:

            #
            P, D, Q = svd(S)

            #
            self.rank_history.append(self.rank)

            return (
                U @ P[:, :self.rank], diag(D[:self.rank]), V @ Q[:, :self.rank]
                )

        if self.N_conserved_basis == 0:

            #
            P, D, Q = svd(S, full_matrices=False)

            #
            maximum(D, 1e-15, out=D)

            #
            rank_augmented = len(D)

            #
            rmax = -1

            #
            total_norm = sqrt(nsum(D**2))

            #
            tol = max(
                self.truncation_tolerance_rel * total_norm,
                self.truncation_tolerance_abs)

            #
            for index in range(rank_augmented):

                #
                residual_energy = sqrt(sum(D[index:]**2))

                #
                if residual_energy < tol:

                    #
                    rmax = index

                    break

            #
            if rmax == -1:

                #
                rmax = rMaxTotal

            #
            rank_limit = self._capacity // 2

            #
            rmax = min(rmax, rMaxTotal)
            rmax = min(rmax, rank_limit)
            rmax = max(rmax, rMinTotal)

            # Update the global rank
            self.rank = rmax
            print('Rank: ', self.rank)

            #
            self.rank_history.append(self.rank)

            #
            U_new = U[:, :rank_augmented] @ P[:, :rmax]
            S_new = diag(D[:rmax])
            V_new = V[:, :rank_augmented] @ Q[:rmax, :].T

            return  U_new, S_new, V_new

        # Conservative truncation
        Khat = U @ S
        Khat_ap, Khat_rem = (
            Khat[:, :self.N_conserved_basis],
            Khat[:, self.N_conserved_basis+1:]) # Splitting Khat into basis required for Ap and remaining vectors
        Vap, Vrem = (
            V[:,:self.N_conserved_basis],
            V[:,self.N_conserved_basis+1:]) # Splitting Khat into basis required for Ap and remaining vectors

        Uhrem, Shrem = qr(Khat_rem)

        P, D, Q = svd(Shrem)

        rmax = -1
        tmp = 0.0

        tol = self.truncation_tolerance_rel * norm(D)

        # Truncating the rank
        for i in range(self.rank - self.N_conserved_basis):

            tmp = sqrt(sum(D[i:]^2))

            if tmp < tol:

                rmax = i + 1
                break

        rmax = min(rmax, rMaxTotal)
        rmax = max(rmax, rMinTotal)

        if rmax == -1:
            rmax = rMaxTotal

        Phat = P[:, :rmax]
        Qhat = Q[:, :rmax]
        sigma_hat = diag(D[1:rmax])

        Q1 = Vrem * Qhat
        Urem = Uhrem * Phat

        V = hstack(Vap, Q1)
        Uap, Sap = qr(Khat_ap)
        U, R2 = qr(hstack(Uap, Urem))

        S = zeros(
            rmax+self.N_conserved_basis, rmax+self.N_conserved_basis)
        S[:self.N_conserved_basis, :self.N_conserved_basis] = Sap
        S[self.N_conserved_basis+1:, self.N_conserved_basis+1:] = sigma_hat
        S = R2 @ S
        self.rank = rmax + self.N_conserved_basis

        #
        self.rank_history.append(self.rank)

        return U, S, V
