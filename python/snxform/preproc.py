"""Flux preprocessing routines.
"""

import os
import numpy as np
from snxform import base_path

from desispec.interpolation import resample_flux

from astropy.table import Table


#- Read in skylines.
skylines = Table.read(os.path.join(base_path, 'etc/skylines.ecsv'))


def rescale_flux(flux: np.ndarray) -> np.ndarray:
    """Rescale flux so that it ranges between 0 and 1.

    Parameters
    ----------
    flux : ndarray
        Input flux array.

    Returns
    -------
    flux_scaled : ndarray
        Flux rescaled to range between 0 and 1.
    """
    if flux.ndim > 1:
        a, b = np.min(flux,axis=1)[:,None], np.max(flux,axis=1)[:,None]
    else:
        a, b = np.min(flux), np.max(flux)

    return (flux - a) / (b - a)


def rebin_flux(wave: np.ndarray, flux: np.ndarray, ivar: np.ndarray=None, z: float=None, minwave: float=3600., maxwave: float=9800., nbins: int=1, log: bool=False, clip: bool=False) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """Rebin differential flux vs wavelength using desispec resample_flux.
    Parameters
    ----------
    wave : ndarray
        Input wavelength; assume units of Angstroms.
    flux : ndarray
        Input differential spectra as a function of wavelength.
    ivar : None or ndarray
        Inverse variance (weight) of spectra vs wavelength.
    z : None, float, or ndarray
        Known or estimated redshift(s) for input spectra.
    minwave : float
        Minimum output wavelength, in units of Angstroms.
    maxwave : float
        Maximum output wavelength, in units of Angstroms.
    nbins : int
        Number of output wavelength bins.
    log : bool
        If true, use logarithmic bins between minwave and maxwave.
    clip : bool
        If true, clip input values below zero before rescaling.
    Returns
    -------
    basewave : ndarray
        Output wavelength, in units of Angstroms.
    fl : ndarray
        Rebinned spectra.
    iv : ndarray
        Rebinned inverse variance.
    """
    #- Choose new binning.
    if log:
        basewave = np.logspace(np.log10(minwave), np.log10(maxwave), nbins)
    else:
        basewave = np.linspace(minwave, maxwave, nbins)

    #- Shift to rest frame (really only used for simulations).
    if z is not None:
        wave = wave/(1+z) if np.isscalar(z) else np.outer(1./(1+z), wave)

    if flux.ndim > 1:
        #- Remove spectra with NaNs and zero flux values.
        mask = np.isnan(flux).any(axis=1) | (np.count_nonzero(flux, axis=1) == 0)
        mask_idx = np.argwhere(mask)
        flux = np.delete(flux, mask_idx, axis=0)
        ivar = np.delete(ivar, mask_idx, axis=0)
        if wave.ndim > 1:
            wave = np.delete(wave, mask_idx, axis=0)

        nspec = len(flux)
        fl = np.zeros((nspec, nbins))
        iv = np.ones((nspec, nbins))
        for i in range(nspec):
            #- Wavelength array may be different for each input flux, e.g., if we shift to the rest frame.
            wave_ = wave[i] if wave.ndim > 1 else wave

            if ivar is not None:
                fl[i], iv[i] = resample_flux(basewave, wave_, flux[i], ivar[i])
            else:
                fl[i] = resample_flux(basewave, wave_, flux[i])
    else:
        resampled = resample_flux(basewave, wave, flux, ivar)
        if ivar is not None:
            fl, iv = resampled
        else:
            fl, iv = resampled, None

    #- Enable clipping of negative values.
    if clip:
        fl = fl.clip(min=0)

    return basewave, fl, iv


def remove_sky_lines(wave: np.ndarray, flux: np.ndarray, ivar: np.ndarray, remove_window: int=2, filter_window: int=10) -> np.ndarray:
    """Remove sky lines (obviously)
        
    Parameters
    ---------- 
    wave : ndarray
        Array of wavelengths 
    flux : ndarray
        Array of flux values
    idxs: ndarray or list
        List of indices corresponding to sky lines in the observer frame
    remove_window : int
        Half-width of window around each sky line to zero out (default=2)
    filter_window: int
        Half-width of window to compute "fill-in" value in sky line, using median.

    Returns
    -------
    newflux: ndarray
        Flux with skylines removed.
    """

    median = np.median(flux)
    stdev = np.ones_like(ivar)
    stdev[ivar > 0] = np.sqrt(1/ivar[ivar > 0])

    newflux = flux.copy()
    idxs = np.asarray([(np.abs(wave - skyline)).argmin() for skyline in skylines['Wavelength']])

    for i in range(flux.shape[0]):
        for idx in idxs:
            if np.any(np.abs(newflux[i, idx-2:idx+3]) >= 3*stdev[i, idx-2:idx+3]):
                newflux[i, idx-remove_window:idx+remove_window+1] = np.median(newflux[i, idx-filter_window:idx+filter_window+1])

    return newflux
