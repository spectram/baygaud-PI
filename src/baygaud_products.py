"""
Functions to convert 2D maps into pickle file and 3D model cubes.

"""

import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
import pickle
import sys
from spectral_cube import SpectralCube
import glob

#ToDo: Write functions to modify headers

def map_to_dict(path_cube, path_classified=None):
    
    #ToDo: Serialize the 2D data arrays
    
    if path_classified == None:
        path_classified=os.path.split(path_cube)[0]+'/baygaud_output.'\
            +os.path.split(path_cube)[1].split(os.extsep)[0]+'/output_merged/classified_intsn0.0_peaksn3.0'
        if os.path.exists(path_classified):
            print('Baygaud output found',path_classified)
        else:
            path_classified=input('Baygaud output not found. Please specify the correct path.')
    
    oghdu=fits.open(path_cube)
    oghdr=oghdu[0].header
    ogwcs=WCS(oghdr)

    channels=np.arange(oghdr['NAXIS3'])
    _, _, channels_wcs = ogwcs.wcs_pix2world(0, 0, channels, 0)

    if u.Unit(oghdr['CUNIT3']) != u.km/u.s:
        channels_wcs *= ogwcs.wcs.cunit[2].to('km/s')
        cdelt3=abs(oghdr['CDELT3']*u.Unit(oghdr['CUNIT3'])).to('km/s').value

    dict_data = {'header':oghdr,'ogdata':oghdu[0].data,'spectral_axis':channels_wcs}

    n_gauss = len(glob.glob(path_classified+"/G0*/"))
    bg = fits.getdata(glob.glob(path_classified+'/G0*g01/*.0.fits')[0])
    data_noise = fits.getdata(glob.glob(path_classified+'/G0*g01/*6.fits')[0])
    dict_data['compmap'] = fits.getdata(glob.glob(path_classified+'/G0*g01/*5.fits')[0])

    dict_data['noise'] = data_noise

    amps   = np.empty(n_gauss, dtype=object)
    vels   = np.empty(n_gauss, dtype=object) 
    vels_chan   = np.empty(n_gauss, dtype=object) 
    disps  = np.empty(n_gauss, dtype=object) 
    tag  = np.empty(n_gauss, dtype=object) 

    for i in range(n_gauss):

        #if os.path.exists(path_classified+'/bulk'):
        bulk_vel = fits.getdata(glob.glob(path_classified+'/bulk/*.bulk{}.3.fits'.format(i+1))[0])
        bulk_psn   = fits.getdata(glob.glob(path_classified+'/bulk/*.bulk{}.7.fits'.format(i+1))[0])
        bulk_disp  = fits.getdata(glob.glob(path_classified+'/bulk/*.bulk{}.2.fits'.format(i+1))[0])

        name_psn   = glob.glob(path_classified+'/G0*g0{}/*7.fits'.format(i+1))[0]
        name_vel   = glob.glob(path_classified+'/G0*g0{}/*3.fits'.format(i+1))[0]
        name_disp  = glob.glob(path_classified+'/G0*g0{}/*2.fits'.format(i+1))[0]
        #name_amp   = glob.glob(path_classified+'/G0*g0{}/*1.fits'.format(i+1))[0] #integrated intensity

        vels[i]   = fits.getdata(name_vel)
        _, _, vels_chan[i] = ogwcs.wcs_world2pix(0, 0, vels[i]*1000, 0)
        disps[i]  = fits.getdata(name_disp)
        #amps[i] = fits.getdata(name_amp) #integrated intensity
        data_psn   = fits.getdata(name_psn)
        amps[i] = data_psn * data_noise
        
        # Check if the component is tagged as bulk
        tag[i]=np.empty(vels[i].shape, dtype=object)
        comp_chk1 = (~np.isnan(vels[i]))
        comp_chk2 = (vels[i]==bulk_vel) # Identifying the bulk component with velocity centroid
        comp_chk3 = (disps[i]==bulk_disp) # Identifying the bulk component with velocity dispersion

        if not np.array_equal(comp_chk3,comp_chk2):
            print('The dispersion and velocity centroid checks are non-identical')

        mask_bulk=np.logical_and(comp_chk1,comp_chk2,comp_chk3)
        mask_non_bulk=np.logical_and(comp_chk1,~comp_chk2,~comp_chk3)

        tag[i] = np.select([mask_bulk, mask_non_bulk], ['bulk', 'non-bulk'], default='')

        testdata=vels[i][mask_bulk]
        
    #if not ((testdata==bulk_vel)):
    #    print('The bulk tags and bulk velocity field do not match')

    dict_data['amp_fit']  = amps
    dict_data['vel_fit']  = vels_chan
    dict_data['disp_fit'] = disps
    dict_data['bg']    = bg
    dict_data['fwhm_fit'] = 2.355*disps/cdelt3
    dict_data['tag'] = tag

    pickle.dump(dict_data, open(path_classified+'/baygaud_decomp.pickle', 'wb'), protocol=2)
    
    
    def dict_to_cubes():
        print('WIP')
        return