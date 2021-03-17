# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import numpy as np
import quaternion
from ..mode_calculations import LLDominantEigenvector, corotating_frame
from .. import jit, Coprecessing, Coorbital, Corotating, Inertial
from .. import sigma, psi4, psi3, psi2, psi1, psi0


def to_corotating_frame(
    self,
    ref_dataType,
    R0=quaternion.one,
    tolerance=1e-12,
    z_alignment_region=None,
    return_omega=False,
    truncate_log_frame=False,
):
    """Transform all waveforms (in place) to a corotating frame.
    Parameters
    ----------
    W: waveform
        Waveform object to be transformed in place
    ref_dataType: {scri.sigma, scri.psi4, scri.psi3, scri.psi2, scri.psi1, scri.psi0}
        Specify which asymptotic quantity to use for setting the corotating frame.
    R0: quaternion [defaults to 1]
        Initial value of frame when integrating angular velocity
    tolerance: float [defaults to 1e-12]
        Absolute tolerance used in integration of angular velocity
    z_alignment_region: None or 2-tuple of floats [defaults to None]
        If not None, the dominant eigenvector of the <LL> matrix is aligned with the z axis,
        averaging over this portion of the data.  The first and second elements of the input are
        considered fractions of the inspiral at which to begin and end the average.  For example,
        (0.1, 0.9) would lead to starting 10% of the time from the first time step to the max norm
        time, and ending at 90% of that time.
    return_omega: bool [defaults to False]
        If True, return a 2-tuple consisting of the waveform in the corotating frame (the usual
        returned object) and the angular-velocity data.  That is frequently also needed, so this is
        just a more efficient way of getting the data.
    truncate_log_frame: bool [defaults to False]
        If True, set bits of log(frame) with lower significance than `tolerance` to zero, and use
        exp(truncated(log(frame))) to rotate the waveform.  Also returns `log_frame` along with the
        waveform (and optionally `omega`)
    """

    ref_data = self.select_data(ref_dataType)
    frame, omega = corotating_frame(
        ref_data, R0=R0, tolerance=tolerance, z_alignment_region=z_alignment_region, return_omega=True
    )
    if truncate_log_frame:
        log_frame = quaternion.as_float_array(np.log(frame))
        power_of_2 = 2 ** int(-np.floor(np.log2(2 * tolerance)))
        log_frame = np.round(log_frame * power_of_2) / power_of_2
        frame = np.exp(quaternion.as_quat_array(log_frame))
    self.rotate_decomposition_basis(frame)
    self.frameType = Corotating
    if return_omega:
        if truncate_log_frame:
            return (self, omega, log_frame)
        else:
            return (self, omega)
    else:
        if truncate_log_frame:
            return (self, log_frame)
        else:
            return self


def to_coprecessing_frame(self, ref_dataType, RoughDirection=np.array([0.0, 0.0, 1.0]), RoughDirectionIndex=None):
    """Transform all waveforms (in place) to a coprecessing frame.
    Parameters
    ----------
    self: waveform
        Waveform object to be transformed in place.
    ref_dataType: {scri.sigma, scri.psi4, scri.psi3, scri.psi2, scri.psi1, scri.psi0}
        Specify which asymptotic quantity to use for setting the coprecessing frame.
    RoughDirection: 3-array [defaults to np.array([0.0, 0.0, 1.0])]
        Vague guess about the preferred initial axis, to choose the sign of the eigenvectors.
    RoughDirectionIndex: int or None [defaults to None]
        Time index at which to apply RoughDirection guess.
    """
    # Decide which waveform quantity to use as a reference for setting the coprecessing frame.
    ref_data = self.select_data(ref_dataType)

    if RoughDirectionIndex is None:
        RoughDirectionIndex = self.n_times // 8
    dpa = LLDominantEigenvector(ref_data, RoughDirection=RoughDirection, RoughDirectionIndex=RoughDirectionIndex)
    R = np.array([quaternion.quaternion.sqrt(-quaternion.quaternion(0, *q).normalized() * quaternion.z) for q in dpa])
    R = quaternion.minimal_rotation(R, self.t, iterations=3)
    self.rotate_decomposition_basis(R)
    self.frameType = Coprecessing
    return self


def to_inertial_frame(self):
    """Undo all frame rotations."""
    self.rotate_decomposition_basis(~self.frame)
    self.frameType = Inertial
    return self


def rotate_physical_system(self, R_phys):
    """Rotate all waveforms in place.
    This just rotates the decomposition basis by the inverse of the input
    rotor(s).  See `rotate_decomposition_basis`.
    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs
    """
    self = self.rotate_decomposition_basis(~R_phys)
    return self  # Probably no return, but just in case...


def rotate_decomposition_basis(self, R_basis):
    """Rotate all waveforms in place.
    This function takes a Waveform object `W` and either a quaternion
    or array of quaternions `R_basis`.  It applies that rotation to
    the decomposition basis of the modes in the Waveform.  The change
    in basis is also recorded in the Waveform's `frame` data.
    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs
    """
    # Rotate all the waveform data
    self.sigma = self.sigma.rotate_decomposition_basis(R_basis)
    self.psi4 = self.psi4.rotate_decomposition_basis(R_basis)
    self.psi3 = self.psi3.rotate_decomposition_basis(R_basis)
    self.psi2 = self.psi2.rotate_decomposition_basis(R_basis)
    self.psi1 = self.psi1.rotate_decomposition_basis(R_basis)
    self.psi0 = self.psi0.rotate_decomposition_basis(R_basis)

    # Record the frame information
    if isinstance(R_basis, (list, np.ndarray)) and len(R_basis) == 1:
        R_basis = R_basis[0]

    if isinstance(R_basis, (list, np.ndarray)):
        # Update the frame data, using right-multiplication
        if self.frame.size:
            if self.frame.shape[0] == 1:
                # Numpy can't currently multiply one element times an array
                self.frame = np.array([self.frame * R for R in R_basis])
            else:
                self.frame = self.frame * R_basis
        else:
            self.frame = np.copy(R_basis)

    if isinstance(R_basis, np.quaternion):
        # Update the frame data, using right-multiplication
        if self.frame.size:
            self.frame = self.frame * R_basis
        else:
            self.frame = np.array([R_basis])

    return self
