"""File I/O routines.
"""

import os
import numpy as np
from snxform import base_path

from desispec.spectra import Spectra
from desispec.io import read_spectra, write_spectra
from desispec.coaddition import coadd_cameras


def read_spectra(fitsfile: str): -> desispec.spectra.Spectra
    """Read in DESI EDR data and apply basic selections. Details of target masking are given in https://desidatamodel.readthedocs.io/en/latest/bitmasks.html.

    Parameters
    ----------
    fitsfile : str
        Path to FITS data containing DESI spectra.
    
    Returns
    -------
    cspectra : desispec.spectra.Spectra
        Spectra object with coadded, selected fluxes.
    """
    #- Read the spectra and coadd the 'b', 'r', and 'z' cameras.
    spectra = read_spectra(fitsfile)
    cspectra = coadd_cameras(spectra)

    #- Access the FIBERMAP and EXP_FIBERMAP to apply selections.
    fmap = cspectra.fibermap
    expfmap = cspectra.exp_fibermap

    #- Select science targets, not SKY fibers.
    select  = (fmap['OBJTYPE'] == 'TGT')

    #- Good fibers have FIBERSTATUS 0 or bit 8 set (MISSINGPOSITION).
    select &= (expfmap['FIBERSTATUS'] == 0) | (expfmap['FIBERSTATUS'] == 1<<3)

    #- Select targets with BGS_ANY bit (bit number 60) set.
    targetbits = np.zeros_like(select, dtype=bool)
    for col in fmap.columns:
        if col.endswith('DESI_TARGET'):
            targetbits |= (fmap[col] & 1<<60 > 0)
    select &= targetbits

    #- Apply the section to the spectra.
    return cspectra[select]

