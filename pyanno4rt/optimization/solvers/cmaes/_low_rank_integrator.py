"""Dynamical low-rank integrator."""

# Authors: Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import diag, shape, hstack, sqrt, zeros
from numpy.linalg import svd, qr, norm

# %% Dynamical low-rank integrator class


class LowRankIntegrator:
    """
    Dynamical low-rank integrator class.

    This class implements ...

    Parameters
    ----------
    ...
    """

    def __init__(
        self,
        name,
        rank,
        truncation_tolerance,
        N_conserved_basis,
        K_step,
        L_step,
        S_step):

        # Get the input attributes
        self.name = name
        self.rank = rank
        self.truncation_tolerance = truncation_tolerance
        self.N_conserved_basis = N_conserved_basis
        self.K_step = K_step
        self.L_step = L_step
        self.S_step = S_step

    def update(
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

        #
        if self.name == "fixedrankBUG":

            #
            return self.fixedrankBUG_onestep(U, S, V, dt)

        #
        elif self.name == "fixedranksymmetricBUG":

            #
            return self.fixedranksymmetricBUG_onestep(U, S, V, dt)

        #
        elif self.name == "fixedaugBUG":

            #
            return self.fixedaugBUG_onestep(U, S, V, dt)

        #
        elif self.name == "augBUG":

            #
            return self.augBUG_onestep(U, S, V, dt)

        #
        elif self.name == "parBUG":

            #
            return self.parBUG_onestep(U, S, V, dt)

        else:

            #
            raise ValueError("The integrator has not been implemented it yet")

    def fixedrankBUG_onestep(
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
        Uhat,_ = qr(K)
        M = Uhat @ U

        L = self.L_step(V @ S.T, U, dt)
        Vhat,_ = qr(L)
        N = Vhat @ V

        U, V = Uhat, Vhat

        S = self.S_step(U, M @ S @ N.T, V, U, M @ S @ N.T, V, dt)

        return U, S, V

    def fixedranksymmetricBUG_onestep(
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

        K = self.K_step(U @ S, U, dt)
        Uhat,_ = qr(K)
        M = Uhat @ U

        U = Uhat

        S = self.S_step(U, M @ S @ M.T, U, U, M @ S @ M.T, U, dt)

        return U, S, V

    def fixedaugBUG_onestep(
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
        Uhat,_ = qr(hstack(K,U))
        M = Uhat @ U

        L = self.L_step(V @ S.T, U, dt)
        Vhat,_ = qr(hstack(L,V))
        N = Vhat @ V

        U, V = Uhat, Vhat

        Shat = self.S_step(U, M @ S @ N.T, V, U, M @ S @ N.T, V, dt)

        U, S, V = self.truncate(Uhat, Shat, Vhat, fixed = True)

        return U, S, V

    def augBUG_onestep(
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

        K = self.K_step(U * S, V, dt)
        Uhat, _ = qr(hstack((K, U)))
        M = Uhat @ U

        L = self.L_step(V * S.T, U, dt)
        Vhat, _ = qr(hstack((L, V)))
        N = Vhat @ V

        U, V = Uhat, Vhat

        Shat = self.S_step(U, M * S @ N.T, V, U, M * S @ N.T, V, dt)

        U, S, V = self.truncate(Uhat, Shat, Vhat)

        return U, S, V

    def parBUG_onestep(
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
        m, r = shape(U)
        n, _ = shape(V)
        rMaxTotal = min(m, n)
        rMinTotal = 2

        if fixed:

            P, D, Q = svd(S)

            return  U @ P[:, :self.rank], D[:self.rank], V @ Q[:, :self.rank]

        else:

            if self.N_conserved_basis == 0:

                P, D, Q = svd(S)

                rmax = -1

                # adaptIndex = 1;

                tmp = 0.0
                tol = self.truncation_tolerance * norm(D)

                for j in range(self.rank):
                    tmp = sqrt(sum(D[j:2*rmax])**2)
                    if tmp < tol:
                        rmax = j + 1
                        break

                # if 2*r was actually not enough move to highest possible rank
                if rmax == -1:
                    rmax = rMaxTotal

                rmax = min(rmax,rMaxTotal)
                rmax = max(rmax,rMinTotal)
                # Updating the global rank to coincide with the updated rank
                self.rank = rmax

                return  U @ P[:, :rmax], D[:rmax], V @ Q[:, :rmax]

            else:

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

                tol = self.truncation_tolerance * norm(D)

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

            return U, S, V

