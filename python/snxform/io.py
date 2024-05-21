"""File I/O routines.
"""

import os
import numpy as np
from snxform import base_path

import desispec
from desispec.spectra import Spectra
from desispec.io import read_spectra, write_spectra
from desispec.coaddition import coadd_cameras

from astropy.table import Table

def read_desi_spectra(specfile: str, redshift: str=None) -> desispec.spectra.Spectra:
    """Read in DESI EDR data and apply basic selections. Details of target masking are given in https://desidatamodel.readthedocs.io/en/latest/bitmasks.html.

    Parameters
    ----------
    specfile : str
        Path to FITS data containing DESI spectra.
    redshift : str or None
        Path to redshift data.
    
    Returns
    -------
    cspectra : desispec.spectra.Spectra
        Spectra object with coadded, selected fluxes.
    """
    #- Read the spectra. If there are separate 'b', 'r', and 'z' data, coadd them.
    spectra = read_spectra(specfile)
    if 'brz' in spectra.wave:
        cspectra = spectra
    else:
        cspectra = coadd_cameras(spectra)

    #- Access the FIBERMAP and EXP_FIBERMAP to apply selections.
    fmap = cspectra.fibermap
    expfmap = cspectra.exp_fibermap

    #- Select science targets, not SKY fibers.
    select  = (fmap['OBJTYPE'] == 'TGT')

    #- Good fibers have FIBERSTATUS 0 or bit 8 set (MISSINGPOSITION).
    #- Be careful to check for multiple exposures in a night.
    expids = np.unique(cspectra.exp_fibermap['EXPID'])
    nexp = expids.shape[0]
    if nexp > 1:
        for expid in expids:
            selexp = expfmap['EXPID'] == expid
            select &= (expfmap['FIBERSTATUS'][selexp] == 0) | (expfmap['FIBERSTATUS'][selexp] == 1<<3)
    else:
        select &= (expfmap['FIBERSTATUS'] == 0) | (expfmap['FIBERSTATUS'] == 1<<3)

    #- Select targets with BGS_ANY bit (bit number 60) set.
    targetbits = np.zeros_like(select, dtype=bool)
    for col in fmap.columns:
        if col.endswith('DESI_TARGET'):
            targetbits |= (fmap[col] & 1<<60 > 0)
    select &= targetbits

    #- If a redshift file is specified, add it as an extra table.
    if redshift is not None:
        zvals = Table.read(redshift, 'REDSHIFTS')
        cspectra.extra_catalog = zvals

    #- Apply the section to the spectra.
    return cspectra[select]

