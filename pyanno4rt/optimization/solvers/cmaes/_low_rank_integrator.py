"""Dynamical low-rank integrator."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import (
    clip, copyto, diag, eye, float64, matmul, maximum, zeros)
from numpy import sum as nsum
from numpy.linalg import norm, pinv, svd
from scipy.linalg import qr

# %% Dynamical low-rank integrator class


class LowRankIntegrator:
    """
    Dynamical low-rank integrator class.

    This class implements update steps for different types of dynamical \
    low-rank integrators.

    Parameters
    ----------
    name : {'fixedBUG', 'fixedsymmetricBUG', 'fixedaugBUG', 'fixedSPDBUG', \
            'augBUG', 'symmetricaugBUG'}
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
            'fixedSPDBUG': self.fixedSPDBUG_step,
            'augBUG': self.augBUG_step,
            'symmetricaugBUG': self.symmetricaugBUG_step
            }
        self.update_func = updates.get(name, self.fixedsymmetricBUG_step)

        # Initialize buffers for the low-rank factors
        self._K = None
        self._Uhat = None
        self._L = None
        self._Vhat = None
        self._M = None
        self._N = None
        self._psi = None

        # Initialize the capacity
        self._capacity = None

        # Initialize the rank history
        self.rank_history = [rank]

    def set_buffers(
            self,
            number_of_variables):
        """
        Initialize the low-rank factor buffers.

        Parameters
        ----------
        number_of_variables : int
            Dimension of the search space (number of decision variables).
        """

        # Get the capacity
        self._capacity = number_of_variables

        # Initialize K
        self._K = zeros(
            (number_of_variables, number_of_variables), order='F',
            dtype=float64)
        self._K[:self.rank, :self.rank] = eye(self.rank, dtype=float64)

        # Initialize Uhat
        self._Uhat = zeros(
            (number_of_variables, number_of_variables), order='F',
            dtype=float64)

        # Check if a symmetric integrator is used
        if 'symmetric' not in self.name.lower():

            # Initialize L
            self._L = zeros(
                (number_of_variables, number_of_variables),
                dtype=float64, order='F')
            self._L[:self.rank, :self.rank] = eye(self.rank, dtype=float64)

            # Initialize Vhat
            self._Vhat = zeros(
                (number_of_variables, number_of_variables),
                dtype=float64, order='F')

            # Initialize N
            self._N = zeros(
                (number_of_variables, number_of_variables), dtype=float64)
            self._N[:self.rank, :self.rank] = eye(self.rank, dtype=float64)

        # Initialize M and N
        self._M = zeros(
            (number_of_variables, number_of_variables), dtype=float64)
        self._M[:self.rank, :self.rank] = eye(self.rank, dtype=float64)

        # Initialize psi
        self._psi = zeros(
            (number_of_variables, number_of_variables), dtype=float64)

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
        rank = U.shape[1]

        #
        K_slice = self._K[:, :rank]
        L_slice = self._L[:, :rank]
        Uhat_slice = self._Uhat[:, :rank]
        Vhat_slice = self._Vhat[:, :rank]
        M_proj = self._M[:rank, :rank]
        N_proj = self._N[:rank, :rank]

        #
        matmul(U, S, out=K_slice)
        matmul(V, S.T, out=L_slice)

        #
        K_updated = self.K_step(K_slice, V, dt)
        L_updated = self.L_step(U, L_slice, dt)

        #
        Uhat, _ = qr(K_updated, mode='economic', check_finite=False)
        copyto(Uhat_slice, Uhat)

        #
        Vhat, _ = qr(L_updated, mode='economic', check_finite=False)
        copyto(Vhat_slice, Vhat)

        #
        matmul(Uhat_slice.T, U, out=M_proj)
        matmul(Vhat_slice.T, V, out=N_proj)

        #
        ext_S = M_proj @ S @ N_proj.T
        Shat = self.S_step(
            Uhat_slice, ext_S, Vhat_slice, None, None, None, dt)

        return Uhat_slice, Shat, Vhat_slice

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
        matmul(U, S, out=K_slice)

        #
        K_updated = self.K_step(K_slice, U, dt)

        #
        Uhat, _ = qr(K_updated, mode='economic', check_finite=False)
        copyto(Uhat_slice, Uhat)

        #
        M_proj = self._M[:rank, :rank]
        matmul(Uhat_slice.T, U, out=M_proj)

        #
        ext_S = M_proj @ S @ M_proj.T
        Shat = self.S_step(
            Uhat_slice, ext_S, Uhat_slice, None, None, None, dt)

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

        #
        max_rank = min(2*rank, self._capacity)

        #
        aug_size = max_rank - rank

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

        #
        if aug_size > 0:

            #
            copyto(self._K[:, rank:max_rank], U[:, :aug_size])

        #
        self.L_step(L_slice, U, dt)

        #
        if aug_size > 0:

            #
            copyto(self._L[:, rank:max_rank], V[:, :aug_size])

        #
        Uhat, _ = qr(K_aug, mode='economic', check_finite=False)
        copyto(Uhat_aug, Uhat)

        #
        Vhat, _ = qr(L_aug, mode='economic', check_finite=False)
        copyto(Vhat_aug, Vhat)

        #
        M_proj = self._M[:max_rank, :rank]
        N_proj = self._N[:max_rank, :rank]
        matmul(Uhat_aug.T, U, out=M_proj)
        matmul(Vhat_aug.T, V, out=N_proj)

        #
        ext_S = M_proj @ S @ N_proj.T
        Shat = self.S_step(
            Uhat_aug, ext_S, Vhat_aug, Uhat_aug, ext_S, Vhat_aug, dt)

        return self.truncate(Uhat_aug, Shat, Vhat_aug, fixed=True)

    def fixedSPDBUG_step(
        self,
        U,
        S,
        _,
        dt):
        """
        Perform a single step of the fixed-rank SPD BUG integrator.
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
        matmul(U, S, out=self._K)

        U_proj = eye() - U @ U.T
        U_proj_had = U_proj * U_proj
        U_proj_had_pinv = pinv(U_proj_had)

        d_psi = U_proj_had_pinv @ diag(U_proj @ U_proj) # Add F here

        #
        self._psi += dt * d_psi

        #
        K_updated = self.K_step(self._K, U, dt)

        K_updated -= dt * (d_psi @ U)

        #
        Uhat, _ = qr(K_updated, mode='economic', check_finite=False)
        copyto(self._Uhat, Uhat)

        #
        matmul(self._Uhat.T, U, out=self._M)

        #
        ext_S = self._M @ S @ self._M.T
        Shat = self.S_step(
            self._Uhat, ext_S, self._Uhat, self._Uhat, None, None, dt)

        Shat -= dt * (self._Uhat.T @ d_psi @ self._Uhat)
        #
        Shat += Shat.T
        Shat *= 0.5

        return self._Uhat, Shat, self._Uhat

    # def isofixedSPDBUG_step(
    #         self,
    #         U,
    #         S,
    #         _,
    #         s,
    #         dt):
    #     """
    #     Perform a single step of the fixed-rank SPD BUG integrator.
    #     Parameters
    #     ----------
    #     ...
    #     Returns
    #     -------
    #     ...
    #     """

    #     #

    #     d,rank = shape(U)
    #     # Evaluate F from the RHS
    #     d_s = (np.linalg.trace(F) - np.linalg.trace(U.T @ F @ U)) / (d - rank)

    #     matmul(U, S, out=self._K)

    #     self._K -= s * U

    #     s += dt * d_s

    #     #
    #     K_updated = self.K_step(self._K, U, dt)

    #     K_updated -= dt * (d_s @ U)

    #     #
    #     Uhat, _ = qr(K_updated, mode='economic', check_finite=False)
    #     copyto(self._Uhat, Uhat)

    #     #
    #     matmul(self._Uhat.T, U, out=self._M)

    #     #
    #     ext_S = self._M @ S @ self._M.T
    #     Shat = self.S_step(
    #         self._Uhat, ext_S, self._Uhat, self._Uhat, None, None, dt)


    #     #
    #     Shat += Shat.T
    #     Shat *= 0.5

    #     return self._Uhat, Shat, self._Uhat

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

        #
        max_rank = min(2*rank, self._capacity)

        #
        aug_size = max_rank - rank

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

        #
        if aug_size > 0:

            #
            copyto(self._K[:, rank:max_rank], U[:, :aug_size])

        #
        self.L_step(L_slice, U, dt)

        #
        if aug_size > 0:

            #
            copyto(self._L[:, rank:max_rank], V[:, :aug_size])

        #
        Uhat, _ = qr(K_aug, mode='economic', check_finite=False)
        copyto(Uhat_aug, Uhat)

        #
        Vhat, _ = qr(L_aug, mode='economic', check_finite=False)
        copyto(Vhat_aug, Vhat)

        #
        M_proj = self._M[:max_rank, :rank]
        N_proj = self._N[:max_rank, :rank]
        matmul(Uhat_aug.T, U, out=M_proj)
        matmul(Vhat_aug.T, V, out=N_proj)

        #
        ext_S = M_proj @ S @ N_proj.T
        Shat = self.S_step(
            Uhat_aug, ext_S, Vhat_aug, Uhat_aug, ext_S, Vhat_aug, dt)

        return self.truncate(Uhat_aug, Shat, Vhat_aug)

    def symmetricaugBUG_step(
            self,
            U,
            S,
            _,
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
        max_rank = min(2*rank, self._capacity)

        #
        aug_size = max_rank - rank

        #
        K_slice = self._K[:, :rank]
        K_aug = self._K[:, :max_rank]
        Uhat_aug = self._Uhat[:, :max_rank]

        #
        matmul(U, S, out=K_slice)

        #
        self.K_step(K_slice, U, dt)

        #
        if aug_size > 0:

            #
            copyto(self._K[:, rank:max_rank], U[:, :aug_size])

        #
        Uhat, _ = qr(K_aug, mode='economic', check_finite=False)
        copyto(Uhat_aug, Uhat)

        #
        M_proj = self._M[:max_rank, :rank]
        matmul(Uhat_aug.T, U, out=M_proj)

        #
        ext_S = M_proj @ S @ M_proj.T
        Shat = self.S_step(
            Uhat_aug, ext_S, Uhat_aug, Uhat_aug, ext_S, Uhat_aug, dt)

        #
        Shat += Shat.T
        Shat *= 0.5

        return self.truncate(Uhat_aug, Shat, Uhat_aug)

    # def SPDaugBUG_step(
    #     self,
    #     U,
    #     S,
    #     V,
    #     psi,
    #     dt):
    #     """
    #     Perform a single step of the low-rank plus diagonal augmented BUG integrator.
    #     Parameters
    #     ----------
    #     ...
    #     Returns
    #     -------
    #     ...
    #     """

    #     #
    #     rank = U.shape[1]
    #     max_rank = 2*rank

    #     #
    #     K_slice = self._K[:, :rank]
    #     K_aug = self._K[:, :max_rank]
    #     Uhat_aug = self._Uhat[:, :max_rank]

    #     #
    #     matmul(U, S, out=K_slice)

    #     U_proj = I - U @ U.T
    #     U_proj_had = U_proj * U_proj
    #     U_proj_had_pinv = np.linalg.pinv(U_proj_had)

    #     d_psi = U_proj_had_pinv @ diag(U_proj @ U_proj) # Add F here

    #     psi = diag(psi + dt * d_psi)
    #     #
    #     self.K_step(K_slice, V, dt)
    #     K_slice -= dt * (d_psi @ U)
    #     copyto(self._K[:, rank:max_rank], U)

    #     #
    #     Uhat, _ = qr(K_aug, mode='economic', check_finite=False) # K_aug is initialized at the beginning but never updated
    #     copyto(self._Uhat[:, :Uhat.shape[1]], Uhat)

    #     #
    #     M_proj = self._M[:max_rank, :rank]
    #     matmul(Uhat_aug.T, U, out=M_proj)

    #     #
    #     ext_S = M_proj @ S @ M_proj.T
    #     Shat = self.S_step(
    #         Uhat_aug, ext_S, Uhat_aug, Uhat_aug, ext_S, Uhat_aug, dt)

    #     #
    #     Shat -= dt * (Uhat_aug.T @ d_psi @ Uhat_aug)
    #     Shat += Shat.T
    #     Shat *= 0.5

    #     return self.truncate(Uhat_aug, Shat, Uhat_aug)

    # def isoSPDaugBUG_step(
    #     self,
    #     U,
    #     S,
    #     V,
    #     s,
    #     dt):
    #     """
    #     Perform a single step of the low-rank plus diagonal augmented BUG integrator.
    #     Parameters
    #     ----------
    #     ...
    #     Returns
    #     -------
    #     ...
    #     """

    #     #
    #     d,rank = shape(U)
    #     max_rank = 2*rank

    #     #
    #     K_slice = self._K[:, :rank]
    #     K_aug = self._K[:, :max_rank]
    #     Uhat_aug = self._Uhat[:, :max_rank]

    #     #
    #     matmul(U, S, out=K_slice)

    #     d_s = U_proj_had_pinv @ diag(U_proj @ U_proj) # Add F here

    #     s += dt * d_s
    #     #
    #     self.K_step(K_slice, V, dt)
    #     K_slice -= dt * (d_s * U)
    #     copyto(self._K[:, rank:max_rank], U)

    #     #
    #     Uhat, _ = qr(K_aug, mode='economic', check_finite=False) # K_aug is initialized at the beginning but never updated
    #     copyto(self._Uhat[:, :Uhat.shape[1]], Uhat)

    #     #
    #     M_proj = self._M[:max_rank, :rank]
    #     matmul(Uhat_aug.T, U, out=M_proj)

    #     #
    #     ext_S = M_proj @ S @ M_proj.T
    #     Shat = self.S_step(
    #         Uhat_aug, ext_S, Uhat_aug, Uhat_aug, ext_S, Uhat_aug, dt)

    #     #
    #     Shat += Shat.T
    #     Shat *= 0.5

    #     return self.truncate(Uhat_aug, Shat, Uhat_aug)

    # def parBUG_step(
    #         self,
    #         U,
    #         S,
    #         V,
    #         dt):
    #     """
    #     .

    #     Parameters
    #     ----------
    #     ...

    #     Returns
    #     -------
    #     ...
    #     """

    #     K = self.K_step(U @ S, V, dt)
    #     Utmp,_ = qr(hstack((U,K)))
    #     Utilde = Utmp[:,self.rank+1:]
    #     Uhat = hstack(U, Utilde)

    #     L = self.L_step(V @ S.T, U, dt)
    #     Vtmp,_ = qr(hstack((V,L)))
    #     Vtilde = Vtmp[:,self.rank+1:]
    #     Vhat = hstack(V, Vtilde)

    #     S = self.S_step(U, S, V, U, S, V, dt)

    #     Shat = zeros(2*self.rank,2*self.rank)
    #     Shat[:self.rank,:self.rank] = S
    #     Shat[:self.rank,:self.rank+1:] = L.T @ Vtilde
    #     Shat[self.rank+1:,1:self.rank] = Utilde.T @ K

    #     U, S, V = self.truncate(Uhat, Shat, Vhat)

    #     return U, S, V

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

        #
        rank_augmented = S.shape[0]
        rMinTotal = 2
        rMaxTotal = min(U.shape[0], V.shape[0], self._capacity)

        #
        P, D, Q = svd(S, full_matrices=False)

        #
        if fixed:

            #
            rmax = self.rank

        #
        elif self.N_conserved_basis == 0:

            #
            maximum(D, 1e-15, out=D)

            #
            total_norm = nsum(D**2)

            #
            tol = max(
                self.truncation_tolerance_rel**2 * total_norm,
                self.truncation_tolerance_abs**2)

            #
            rmax = rank_augmented

            #
            for index in range(rank_augmented):

                #
                residual_energy = sum(D[index:]**2)

                #
                if residual_energy < tol:

                    #
                    rmax = index

                    break

        #
        rmax = clip(rmax, rMinTotal, rMaxTotal)

        # Update the global rank
        self.rank = int(rmax)
        print('Rank: ', self.rank)

        #
        self.rank_history.append(self.rank)

        #
        U_new = U @ P[:, :self.rank]
        S_new = diag(D[:self.rank])
        V_new = V @ Q[:self.rank, :].T

        return  U_new, S_new, V_new

        # # Conservative truncation
        # Khat = U @ S
        # Khat_ap, Khat_rem = (
        #     Khat[:, :self.N_conserved_basis],
        #     Khat[:, self.N_conserved_basis+1:]) # Splitting Khat into basis required for Ap and remaining vectors
        # Vap, Vrem = (
        #     V[:,:self.N_conserved_basis],
        #     V[:,self.N_conserved_basis+1:]) # Splitting Khat into basis required for Ap and remaining vectors

        # Uhrem, Shrem = qr(Khat_rem)

        # P, D, Q = svd(Shrem)

        # rmax = -1
        # tmp = 0.0

        # tol = self.truncation_tolerance_rel * norm(D)

        # # Truncating the rank
        # for i in range(self.rank - self.N_conserved_basis):

        #     tmp = sqrt(sum(D[i:]**2))

        #     if tmp < tol:

        #         rmax = i + 1
        #         break

        # rmax = min(rmax, rMaxTotal)
        # rmax = max(rmax, rMinTotal)

        # if rmax == -1:
        #     rmax = rMaxTotal

        # Phat = P[:, :rmax]
        # Qhat = Q[:, :rmax]
        # sigma_hat = diag(D[1:rmax])

        # Q1 = Vrem * Qhat
        # Urem = Uhrem * Phat

        # V = hstack(Vap, Q1)
        # Uap, Sap = qr(Khat_ap)
        # U, R2 = qr(hstack(Uap, Urem))

        # S = zeros(
        #     rmax+self.N_conserved_basis, rmax+self.N_conserved_basis)
        # S[:self.N_conserved_basis, :self.N_conserved_basis] = Sap
        # S[self.N_conserved_basis+1:, self.N_conserved_basis+1:] = sigma_hat
        # S = R2 @ S
        # self.rank = rmax + self.N_conserved_basis

        # #
        # self.rank_history.append(self.rank)

        # return U, S, V
