import pandas as pd
import numpy as np

import os
from os import path
import astropy.io.fits as fits

import warnings
from scipy.interpolate import LSQUnivariateSpline
from pydl.pydlutils import bspline

import sys

TESS_TSOP301_SECTOR_PREFIXES = {
    1: "2018191215100",
    2: "2018220103459",
    3: "2018248225659",
    4: "2018277033859",
}

TESS_TSOP301_DV_SECTOR_PREFIXES ={
    1:["tess2018192011050-s0001-s0001-","-00022_dvt.fits.gz"],
    2:["tess2018220134050-s0002-s0002-","-00028_dvt.fits.gz"],
    3:["tess2018249021050-s0003-s0003-","-00033_dvt.fits.gz"],
    4:["tess2018277071050-s0004-s0004-","-00038_dvt.fits.gz"],
    99:["tess2018192011050-s0001-s0004-","-00055_dvt.fits.gz"]
}

def tess_filenames(base_dir, tic_id, sector=None, simrun='TSOP-301-4-sector-run',use_dv=False):

    """
    Returns the light curve filenames for a TESS target star.

    This function assumes the directory structure of GCP uploaded data as of July 24, 2018

    Args:
        base_dir: Base directory containing TESS data on GPC (currently /mnt/disks/data/tess/lilithSimulatedData)
        simrun: name of lilith simulated run (TSOP-301-4-sector-run, ETE-6/, etc.)
        tess_id: TIC id of target star. May be an int or a possibly zero-padded string.

    Returns:
        A list of filenames with dv filename first and pdc filenames after

    Example:

        base_dir = "/mnt/disks/data/tess/lilithSimulatedData"
        simrun = "TSOP-301-4-sector-run"
        tic_id = 270282215


    """
    # get path for TSOP-310

    if simrun == "TSOP-301-4-sector-run":

        if sector is None or sector == 99 or sector=='s1-s4' or sector=='allsectors-99':
            sector=99
        elif type(sector)==str:
            sector=int(sector[-1])
        else:
            sector=int(sector)
        if use_dv:
            #Using DV product lightcurves
            sector_prefixes = TESS_TSOP301_DV_SECTOR_PREFIXES
            if sector is None or sector == 99 or sector=='s1-s4' or sector=='allsectors-99':
                sector= 99
            prefix = sector_prefixes[sector]
            filenames = [os.path.join(base_dir, simrun, 'sector-' + str(sector),'exports','dv-time-series',prefix[0]+str(int(tic_id)).zfill(16)+prefix[1])]
        else:
            filenames=['None']

        #Using more raw lightcurves
        sector_prefixes = TESS_TSOP301_SECTOR_PREFIXES
        if sector==99:
            sectors=[1,2,3,4]
        else:
            sectors=[sector]
        for sector in sectors:
            sector_prefix = sector_prefixes[sector]
            tic_id = "%.16d" % int(tic_id)
            prefix = "tess" + sector_prefix + "-s000" + str(sector) + "-"
            suffix = "-000" + str(sector) + "-a_lc.fits.gz"
            filename = os.path.join(base_dir, simrun, 'sector-' + str(sector),'exports', 'light-curve', prefix + tic_id + suffix)
            if os.path.isfile(filename):
                filenames.append(filename)

    # other simruns not yet supported
    else:

        print('simrun not yet supported')
        return None

    return filenames


def read_tess_light_curve(filenames, light_curve_extension="LIGHTCURVE", invert=False,use_dv=False,tce=None):

    #print(filenames)
    #print(light_curve_extension)
    #print(invert)

    """
    Reads time and flux measurements for a TESS target star.
    Args:
        filenames: A list of .fits files containing time and flux measurements.
        light_curve_extension: Name of the HDU 1 extension containing light curves.
        invert: Whether to invert the flux measurements by multiplying by -1.
    Returns:
        all_time: A list of numpy arrays; the time values of the light curve.
        all_flux: A list of numpy arrays corresponding to the time arrays in
                all_time.
    """

    all_time = []
    all_flux = []
    all_centroid = []
    all_valid_indices=[]
    if use_dv:
        #Getting DV flux and times:
        dv_fname=filenames[0]#[fi for fi in filenames if '_dvt' in fi][0]
        #print("dv_fname:",dv_fname)
        with fits.open(dv_fname) as hdu_list:
            #Finds nearest epoch and period
            dvepochs=np.array([hdu_list[n].header['TEPOCH'] for n in range(1,len(hdu_list)-1)])
            dvperiods=np.array([hdu_list[n].header['TPERIOD'] for n in range(1,len(hdu_list)-1)])
            test_epcs=np.arange(tce['tce_time0bk']-50*tce['tce_period'], tce['tce_time0bk']+50*tce['tce_period'], tce['tce_period'])
            nrst_epcs=np.array([test_epcs[np.argmin(abs(test_epcs-hdu_list[n].header['TEPOCH']))] for n in range(1,len(hdu_list)-1)])
            idx=np.argmin((abs(tce['tce_period']-dvperiods)/tce['tce_period'])**2+(abs(nrst_epcs-dvepochs)/tce['tce_period'])**2)
            #print('N_nans, white:',np.sum(np.isnan(hdu_list[idx+1].data['LC_WHITE'])), 'init:',np.sum(np.isnan(hdu_list[idx+1].data['LC_INIT'])), 'detrend:',np.sum(np.isnan(hdu_list[idx+1].data['LC_DETREND'])))
            
            #Accesses time and whitened flux lightcurve arrays:
            dvtime=hdu_list[idx+1].data['TIME']
            dvflux=1+hdu_list[idx+1].data['LC_DETREND'] if np.sum(np.isnan(hdu_list[idx+1].data['LC_DETREND']))!=len(dvtime) else 1+hdu_list[idx+1].data['LC_INIT']
        
        #For -99 sectors, the times are always 81195 points long, but get zeroed out for sectors that are not present.
        dvflux=dvflux[dvtime!=0.0]
        dvtime=dvtime[dvtime!=0.0]
    #Getting non-DV timeseries:
    for filename in filenames[1:]:
        with fits.open(filename) as hdu_list:
            light_curve = hdu_list[light_curve_extension].data
            time = np.array(light_curve.TIME)
            flux = np.array(light_curve.PDCSAP_FLUX)
            ## proper 2d centroids
        if np.sum(~np.isnan(light_curve['PSF_CENTR1']))>0:
            centroid =np.column_stack((light_curve['PSF_CENTR1'],light_curve['PSF_CENTR2']))
        else:
            centroid = np.column_stack((light_curve['MOM_CENTR1'], light_curve['MOM_CENTR2']))
            #print("USING MOM CENTROIDS", np.sum(np.isnan(centroid)),np.shape(centroid))
        #from IPython import embed; embed()

        # Remove zero flux values
        all_valid_indices.append(np.isfinite(flux)*(~np.isnan(flux))*(flux != 0.0))
        all_time.append(time)
        all_flux.append(flux)
        all_centroid.append(centroid)
    all_time=np.hstack((all_time))
    all_flux=np.hstack((all_flux))
    all_centroid=np.vstack((all_centroid))
    all_valid_indices=np.hstack((all_valid_indices))
    if use_dv and np.sum(np.isnan(dvflux))<0.5*len(dvtime):
        #Using DV lightcurve to cut transits from centroids and all_time ONLY if dvflux array isnt half nans.
        #print('flux:',len(dvflux),len(all_flux),'| centroid:',np.shape(all_centroid),'| time:',len(dvtime),len(all_time))
        dv_valid_indices=np.isfinite(dvflux)*(~np.isnan(dvflux))*(dvflux != 0.0)
        #print('all',np.sum(all_valid_indices),'dv',np.sum(dv_valid_indices))
        if len(all_time)!=len(dvtime) or len(all_flux)!=len(dvflux):
            #Finding the times that PDC and DV share
            #print('Finding the times that PDC and DV share... problem?')
            pdcbool=np.in1d(np.round(all_time/np.nanmedian(np.diff(all_time))),np.round(dvtime/np.nanmedian(np.diff(all_time))))
            dvbool=np.in1d(np.round(dvtime/np.nanmedian(np.diff(all_time))), np.round(all_time/np.nanmedian(np.diff(all_time))))
            #Cutting any times that are not shared in both:
            all_time,all_flux=all_time[pdcbool],all_flux[pdcbool]
            #print(np.shape(all_valid_indices))
            all_centroid,all_valid_indices=all_centroid[pdcbool], all_valid_indices[pdcbool]
            #print(np.shape(all_valid_indices))
            dvtime,dvflux,dv_valid_indices=dvtime[dvbool],dvflux[dvbool],dv_valid_indices[dvbool]
        all_flux=np.copy(dvflux)
        all_valid_indices=all_valid_indices+dv_valid_indices
        
    all_time = all_time[all_valid_indices]
    all_flux = all_flux[all_valid_indices]
    all_centroid = all_centroid[all_valid_indices]
    #print("Removed flux nans:", np.sum(np.isnan(all_centroid)),np.shape(all_centroid),all_centroid[:5,:])
    
    #print('flux:',len(all_flux),'| centroid:',np.shape(all_centroid),'| time:',len(all_time))

    if invert:
        flux *= -1

    return all_time, all_flux, all_centroid, all_valid_indices

def process_light_curve(time, flux, centr, QUICK_SPLINE=True, only_cent=False):
    """Removes low-frequency variability from a light curve.

    Args:
        all_time: A list of numpy arrays; the time values of the raw light curve.
        all_flux: A list of numpy arrays corresponding to the time arrays in
                all_time.

    Returns:
        time: 1D NumPy array; the time values of the light curve.
        flux: 1D NumPy array; the normalized flux values of the light curve.
    """
    # Split on gaps.
    if not only_cent:
        #FLUX SPLINE FITTING (only when not only_cent)
        all_time, all_flux = split(time, flux, gap_width=0.5)
        
        # Fit a piecewise-cubic spline with default arguments.
        spline = fit_kepler_spline(all_time, all_flux, verbose=False, QUICK_SPLINE=QUICK_SPLINE)[0]
        
        # Concatenate the piecewise light curve and spline.
        assert len(np.concatenate(all_time)) == len(time)

        flux = np.concatenate(all_flux)
        spline = np.concatenate(spline)
        
        finite_i = np.isfinite(spline)
        
        '''
        # In rare cases the piecewise spline contains NaNs in places the spline could
        # not be fit. We can't normalize those points if the spline isn't defined
        # there. Instead we just remove them.
        finite_i = np.isfinite(spline)
        if not np.all(finite_i):
            time = time[finite_i]
            flux = flux[finite_i]
            spline = spline[finite_i]
        '''
            
        # "Flatten" the light curve (remove low-frequency variability) by dividing by
        # the spline.
        flux /= spline
    else:
        finite_i=np.tile(True,len(time))

    #CENTROID SPLINE FITTING (only when not only_cent)
          
    all_time_cent, all_centr_x = split(time, np.nan_to_num(centr[:,0],np.nanmedian(centr[:,0])), gap_width=0.5)
    all_time_cent, all_centr_y = split(time, np.nan_to_num(centr[:,1],np.nanmedian(centr[:,1])), gap_width=0.5)

    #if not only_cent:
    #from IPython import embed;embed()
                             
    spline_1 = fit_kepler_spline(all_time_cent, all_centr_x, verbose=False, QUICK_SPLINE=QUICK_SPLINE)[0]
    spline_2 = fit_kepler_spline(all_time_cent, all_centr_y, verbose=False, QUICK_SPLINE=QUICK_SPLINE)[0]
    spline_1 = np.concatenate(spline_1)
    spline_2 = np.concatenate(spline_2)
    all_centr = np.column_stack((np.concatenate(all_centr_x),np.concatenate(all_centr_y)))
                                 
    finite_i = ~np.isnan(spline_1)*np.isfinite(spline_1)*~np.isnan(spline_2)*np.isfinite(spline_2)*finite_i
    if not np.all(finite_i):
        time = time[finite_i]
        flux = flux[finite_i]
        spline_1 = spline_1[finite_i]
        spline_2 = spline_2[finite_i]
        all_centr = all_centr[finite_i]
    
    #Flattening
    all_centr[:,0] -= spline_1
    all_centr[:,1] -= spline_2

    # distance from center
    all_centr = np.sqrt(all_centr[:,0]**2 + all_centr[:,1]**2 )

    return time, flux, all_centr

def phase_fold_and_sort_light_curve(time, flux, centroid, period, t0,cut_anoms=True):
    """Phase folds a light curve and sorts by ascending time.
tse_fold_and_sort_light_curve
    Args:
        time: 1D NumPy array of time values.
        flux: 1D NumPy array of flux values.
        period: A positive real scalar; the period to fold over.
        t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
        folded_time: 1D NumPy array of phase folded time values in
                [-period / 2, period / 2), where 0 corresponds to t0 in the original
                time array. Values are sorted in ascending order.
        folded_flux: 1D NumPy array. Values are the same as the original input
                array, but sorted by folded_time.
    """
    # Phase fold time
    time=np.array(time).ravel()
    time=fix_harmonic_period(time, period)
    time = phase_fold_time(time, period, t0)
    #print(type(flux))
    #print(flux.shape)
    # Sort by ascending time.
    sorted_i = np.argsort(time)
    time = time[sorted_i]
    
    if np.shape(flux)[0]==1:
        flux=flux[0]
    flux=np.array(flux).ravel()
    flux = flux[sorted_i]
    
    if np.shape(centroid)[0]==1:
        centroid=centroid[0]
        
    centroid = np.array(centroid)
    centroid=centroid[sorted_i]
        
    if cut_anoms:
        nonaninf=(time/time==1.0)*(flux/flux==1.0)*CutAnomDiff(flux)
    else:
        nonaninf=(time/time==1.0)*(flux/flux==1.0)
    #centroids aren't SO important that nans should kill it. Lets just make these zeros
    #centroid=np.nan_to_num(centroid)
        
    return time[nonaninf], flux[nonaninf], centroid[nonaninf]


def fix_harmonic_period(time, period):
    #This function fixes periods which have a harmony with (ie are an integer multiple of) the cadence, which is a problem for folding short-period lightcurves.
    cad=np.nanmedian(np.diff(time))
    #Catching periods which are integer multiple aliases of the cadence, which results in no points in bins.
    #Ncads=len(time)
    Npers=np.floor((time[-1]-time[0])/period) #min number per bundle
    #print("cadence/total_periods",(cad/Npers)," = threshold proximty to harmonic with N<",MinNumBundlesLocal,"required for local")
    NumBundles=np.round(period/cad) #
    proximity_to_harmonic=abs(1-(period/cad)/NumBundles)

    #print("HARMONIC PROXIMITY:"<Plug>PeepOpenroximity_to_harmonic,"from N=",NumBundles,"with spread"<Plug>PeepOpenroximity_to_harmonic*NumBundles*Npers,"<2?")
    if NumBundles<2000 and proximity_to_harmonic*NumBundles*Npers<1.66:
        #Where the spread of the nearest harmonic has less than an overlap of 2, making time normally distributed
        time=np.random.normal(time,0.3*cad)
    return time

def CutAnomDiff(flux,thresh=4.35):
        #Uses differences between points to establish anomalies.
        #Only removes single points with differences to both neighbouring points greater than threshold above median difference (ie ~rms)
        #Fast: 0.05s for 1 million-point array.
        #Must be nan-cut first
        diffarr=np.vstack((np.diff(flux[1:]),np.diff(flux[:-1])))
        diffarr/=np.nanmedian(abs(diffarr[0,:]))
        anoms=np.hstack((True,((diffarr[0,:]*diffarr[1,:])>0)+(abs(diffarr[0,:])<thresh)+(abs(diffarr[1,:])<thresh),True))*np.isfinite(flux)
        return anoms


def phase_fold_time(time, period, t0):
    """Creates a phase-folded time vector.

    result[i] is the unique number in [-period / 2, period / 2)
    such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.

    Args:
        time: 1D numpy array of time values.
        period: A positive real scalar; the period to fold over.
        t0: The center of the resulting folded vector; this value is mapped to 0.

    Returns:
        A 1D numpy array.
    """
    half_period = period / 2
    result = np.mod(time + (half_period - t0), period)
    result -= half_period
    return result


#######################~~~~~~~~ SPLINE STUFF: ~~~~~~~###########################


def split(all_time, all_flux, gap_width=0.75):
    """Splits a light curve on discontinuities (gaps).

    This function accepts a light curve that is either a single segment, or is
    piecewise defined (e.g. split by quarter breaks or gaps in the in the data).

    Args:
        all_time: Numpy array or sequence of numpy arrays; each is a sequence of
                time values.
        all_flux: Numpy array or sequence of numpy arrays; each is a sequence of
                flux values of the corresponding time array.
        gap_width: Minimum gap size (in time units) for a split.

    Returns:
        out_time: List of numpy arrays; the split time arrays.
        out_flux: List of numpy arrays; the split flux arrays.
    """
    # Handle single-segment inputs.
    if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
        all_time = [all_time]
        all_flux = [all_flux]

    out_time = []
    out_flux = []
    for time, flux in zip(all_time, all_flux):
        start = 0
        for end in range(1, len(time) + 1):
            # Choose the largest endpoint such that time[start:end] has no gaps.
            if end == len(time) or time[end] - time[end - 1] > gap_width:
                out_time.append(time[start:end])
                out_flux.append(flux[start:end])
                start = end

    return out_time, out_flux



class InsufficientPointsError(Exception):
    """Indicates that insufficient points were available for spline fitting."""
    pass


class SplineError(Exception):
    """Indicates an error in the underlying spline-fitting implementation."""
    pass



def kepler_spline(time, flux, bkspace=1.5, maxiter=5, outlier_cut=3, QUICK_SPLINE=False):
    """Computes a best-fit spline curve for a light curve segment.

    The spline is fit using an iterative process to remove outliers that may cause
    the spline to be "pulled" by discrepent points. In each iteration the spline
    is fit, and if there are any points where the absolute deviation from the
    median residual is at least 3*sigma (where sigma is a robust estimate of the
    standard deviation of the residuals), those points are removed and the spline
    is re-fit.

    Args:
        time: Numpy array; the time values of the light curve.
        flux: Numpy array; the flux (brightness) values of the light curve.
        bkspace: Spline break point spacing in time units.
        maxiter: Maximum number of attempts to fit the spline after removing badly
                fit points.
        outlier_cut: The maximum number of standard deviations from the median
                spline residual before a point is considered an outlier.

    Returns:
        spline: The values of the fitted spline corresponding to the input time
                values.
        mask: Boolean mask indicating the points used to fit the final spline.

    Raises:
        InsufficientPointsError: If there were insufficient points (after removing
                outliers) for spline fitting.
        SplineError: If the spline could not be fit, for example if the breakpoint
                spacing is too small.
    """
    if len(time) < 4:
        raise InsufficientPointsError(
                "Cannot fit a spline on less than 4 points. Got %d points." % len(time))

    # Rescale time into [0, 1].
    t_min = np.min(time)
    t_max = np.max(time)
    time = (time - t_min) / (t_max - t_min)
    bkspace /= (t_max - t_min)    # Rescale bucket spacing.

    if QUICK_SPLINE:
        # calculate knots of quick spline
        nknot = int((time[-1] - time[0]) / bkspace)

        if nknot == 0:
            nknot = 1

        knots = np.linspace(time[1], time[-2], nknot)


        try:
            k_min = np.min(knots)
            k_max = np.max(knots)
        except:
            from IPython import embed; embed()

        knots = (knots-k_min) / (k_max-k_min)


        knots = knots[1:-2]

    # Values of the best fitting spline evaluated at the time points.
    spline = None

    # Mask indicating the points used to fit the spline.
    mask = None

    for _ in range(maxiter):
        if spline is None:
            mask = np.ones_like(time, dtype=np.bool)    # Try to fit all points.
        else:
            # Choose points where the absolute deviation from the median residual is
            # less than outlier_cut*sigma, where sigma is a robust estimate of the
            # standard deviation of the residuals from the previous spline.
            residuals = flux - spline
            new_mask = robust_mean(residuals, cut=outlier_cut)[2]

            if np.all(new_mask == mask):
                break    # Spline converged.

            mask = new_mask

        if np.sum(mask) < 4:
            # Fewer than 4 points after removing outliers. We could plausibly return
            # the spline from the previous iteration because it was fit with at least
            # 4 points. However, since the outliers were such a significant fraction
            # of the curve, the spline from the previous iteration is probably junk,
            # and we consider this a fatal error.
            raise InsufficientPointsError(
                    "Cannot fit a spline on less than 4 points. After removing "
                    "outliers, got %d points." % np.sum(mask))

        try:
            with warnings.catch_warnings():
                # Suppress warning messages printed by pydlutils.bspline. Instead we
                # catch any exception and raise a more informative error.
                warnings.simplefilter("ignore")

                # Fit the spline on non-outlier points.
                if QUICK_SPLINE:
                    #try:
                    curve = LSQUnivariateSpline(time[mask], flux[mask], knots, k=3)
                    #except:
                    #    from IPython import embed; embed()
                else:
                    curve = bspline.iterfit(time[mask], flux[mask], bkspace=bkspace)[0]



            # Evaluate spline at the time points.
            if QUICK_SPLINE:
                spline = curve(time)
            else:
                spline = curve.value(time)[0]

            #spline = np.copy(flux)

        except (IndexError, TypeError, ValueError) as e:
            raise SplineError(
                    "Fitting spline failed with error: '%s'. This might be caused by the "
                    "breakpoint spacing being too small, and/or there being insufficient "
                    "points to fit the spline in one of the intervals." % e)

    return spline, mask


def fit_kepler_spline(all_time,
                                            all_flux,
                                            bkspace_min=0.5,
                                            #bkspace_max=20,
                                            bkspace_max=15,
                                            bkspace_num=20,
                                            maxiter=5,
                                            penalty_coeff=1.0,
                                            verbose=True,
                                            QUICK_SPLINE=False):
    """Fits a Kepler spline with logarithmically-sampled breakpoint spacings.

    Args:
        all_time: List of 1D numpy arrays; the time values of the light curve.
        all_flux: List of 1D numpy arrays; the flux values of the light curve.
        bkspace_min: Minimum breakpoint spacing to try.
        bkspace_max: Maximum breakpoint spacing to try.
        bkspace_num: Number of breakpoint spacings to try.
        maxiter: Maximum number of attempts to fit each spline after removing badly
                fit points.
        penalty_coeff: Coefficient of the penalty term for using more parameters in
                the Bayesian Information Criterion. Decreasing this value will allow
                more parameters to be used (i.e. smaller break-point spacing), and
                vice-versa.
        verbose: Whether to log individual spline errors. Note that if bkspaces
                contains many values (particularly small ones) then this may cause
                logging pollution if calling this function for many light curves.

    Returns:
        spline: List of numpy arrays; values of the best-fit spline corresponding to
                to the input flux arrays.
        metadata: Object containing metadata about the spline fit.
    """

    ## account for when default bkspace_max > length of total lightcurve
    #time_span = all_time[0][-1] - all_time[0][0]
    #if time_span < bkspace_max:
    #    bkspace_max = time_span

    # Logarithmically sample bkspace_num candidate break point spacings between
    # bkspace_min and bkspace_max.
    bkspaces = np.logspace(
            np.log10(bkspace_min), np.log10(bkspace_max), num=bkspace_num)

    return choose_kepler_spline(
            all_time,
            all_flux,
            bkspaces,
            maxiter=maxiter,
            penalty_coeff=penalty_coeff,
            verbose=verbose, QUICK_SPLINE=QUICK_SPLINE)


class SplineMetadata(object):
    """Metadata about a spline fit.

    Attributes:
        light_curve_mask: List of boolean numpy arrays indicating which points in
                the light curve were used to fit the best-fit spline.
        bkspace: The break-point spacing used for the best-fit spline.
        bad_bkspaces: List of break-point spacing values that failed.
        likelihood_term: The likelihood term of the Bayesian Information Criterion;
                -2*ln(L), where L is the likelihood of the data given the model.
        penalty_term: The penalty term for the number of parameters in the
                Bayesian Information Criterion.
        bic: The value of the Bayesian Information Criterion; equal to
                likelihood_term + penalty_coeff * penalty_term.
    """

    def __init__(self):
        self.light_curve_mask = None
        self.bkspace = None
        self.bad_bkspaces = []
        self.likelihood_term = None
        self.penalty_term = None
        self.bic = None

def choose_kepler_spline(all_time,
                                                 all_flux,
                                                 bkspaces,
                                                 maxiter=5,
                                                 penalty_coeff=1.0,
                                                 verbose=True, QUICK_SPLINE=False):
    """Computes the best-fit Kepler spline across a break-point spacings.

    Some Kepler light curves have low-frequency variability, while others have
    very high-frequency variability (e.g. due to rapid rotation). Therefore, it is
    suboptimal to use the same break-point spacing for every star. This function
    computes the best-fit spline by fitting splines with different break-point
    spacings, calculating the Bayesian Information Criterion (BIC) for each
    spline, and choosing the break-point spacing that minimizes the BIC.

    This function assumes a piecewise light curve, that is, a light curve that is
    divided into different segments (e.g. split by quarter breaks or gaps in the
    in the data). A separate spline is fit for each segment.

    Args:
        all_time: List of 1D numpy arrays; the time values of the light curve.
        all_flux: List of 1D numpy arrays; the flux values of the light curve.
        bkspaces: List of break-point spacings to try.
        maxiter: Maximum number of attempts to fit each spline after removing badly
                fit points.
        penalty_coeff: Coefficient of the penalty term for using more parameters in
                the Bayesian Information Criterion. Decreasing this value will allow
                more parameters to be used (i.e. smaller break-point spacing), and
                vice-versa.
        verbose: Whether to log individual spline errors. Note that if bkspaces
                contains many values (particularly small ones) then this may cause
                logging pollution if calling this function for many light curves.

    Returns:
        spline: List of numpy arrays; values of the best-fit spline corresponding to
                to the input flux arrays.
        metadata: Object containing metadata about the spline fit.
    """
    # Initialize outputs.
    best_spline = None
    metadata = SplineMetadata()

    # Compute the assumed standard deviation of Gaussian white noise about the
    # spline model. We assume that each flux value f[i] is a Gaussian random
    # variable f[i] ~ N(s[i], sigma^2), where s is the value of the true spline
    # model and sigma is the constant standard deviation for all flux values.
    # Moreover, we assume that s[i] ~= s[i+1]. Therefore,
    # (f[i+1] - f[i]) / sqrt(2) ~ N(0, sigma^2).
    scaled_diffs = [np.diff(f) / np.sqrt(2) for f in all_flux]
    scaled_diffs = np.concatenate(scaled_diffs) if scaled_diffs else np.array([])
    if not scaled_diffs.size:
        best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
        metadata.light_curve_mask = [
                np.zeros_like(f, dtype=np.bool) for f in all_flux
        ]
        return best_spline, metadata

    # Compute the median absoute deviation as a robust estimate of sigma. The
    # conversion factor of 1.48 takes the median absolute deviation to the
    # standard deviation of a normal distribution. See, e.g.
    # https://www.mathworks.com/help/stats/mad.html.
    sigma = np.median(np.abs(scaled_diffs)) * 1.48

    for bkspace in bkspaces:
        nparams = 0    # Total number of free parameters in the piecewise spline.
        npoints = 0    # Total number of data points used to fit the piecewise spline.
        ssr = 0    # Sum of squared residuals between the model and the spline.

        spline = []
        light_curve_mask = []
        bad_bkspace = False    # Indicates that the current bkspace should be skipped.
        for time, flux in zip(all_time, all_flux):
            # Fit B-spline to this light-curve segment.
            try:
                spline_piece, mask = kepler_spline(
                        time, flux, bkspace=bkspace, maxiter=maxiter, QUICK_SPLINE=QUICK_SPLINE)
            except InsufficientPointsError as e:
                # It's expected to occasionally see intervals with insufficient points,
                # especially if periodic signals have been removed from the light curve.
                # Skip this interval, but continue fitting the spline.
                if verbose:
                    warnings.warn(str(e))
                spline.append(np.array([np.nan] * len(flux)))
                light_curve_mask.append(np.zeros_like(flux, dtype=np.bool))
                continue
            except SplineError as e:
                # It's expected to get a SplineError occasionally for small values of
                # bkspace. Skip this bkspace.
                if verbose:
                    warnings.warn("Bad bkspace %.4f: %s" % (bkspace, e))
                metadata.bad_bkspaces.append(bkspace)
                bad_bkspace = True
                break

            spline.append(spline_piece)
            light_curve_mask.append(mask)

            # Accumulate the number of free parameters.
            total_time = np.max(time) - np.min(time)
            nknots = int(total_time / bkspace) + 1    # From the bspline implementation.
            nparams += nknots + 3 - 1    # number of knots + degree of spline - 1

            # Accumulate the number of points and the squared residuals.
            npoints += np.sum(mask)
            ssr += np.sum((flux[mask] - spline_piece[mask])**2)

        if bad_bkspace or not npoints:
            continue

        # The following term is -2*ln(L), where L is the likelihood of the data
        # given the model, under the assumption that the model errors are iid
        # Gaussian with mean 0 and standard deviation sigma.
        likelihood_term = npoints * np.log(2 * np.pi * sigma**2) + ssr / sigma**2

        # Penalty term for the number of parameters used to fit the model.
        penalty_term = nparams * np.log(npoints)

        # Bayesian information criterion.
        bic = likelihood_term + penalty_coeff * penalty_term

        if best_spline is None or bic < metadata.bic:
            best_spline = spline
            metadata.light_curve_mask = light_curve_mask
            metadata.bkspace = bkspace
            metadata.likelihood_term = likelihood_term
            metadata.penalty_term = penalty_term
            metadata.bic = bic

    if best_spline is None:
        # All bkspaces resulted in a SplineError, or all light curve intervals had
        # insufficient points.
        best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
        metadata.light_curve_mask = [
                np.zeros_like(f, dtype=np.bool) for f in all_flux
        ]

    return best_spline, metadata



def robust_mean(y, cut):
    """Computes a robust mean estimate in the presence of outliers.

    Args:
        y: 1D numpy array. Assumed to be normally distributed with outliers.
        cut: Points more than this number of standard deviations from the median are
                ignored.

    Returns:
        mean: A robust estimate of the mean of y.
        mean_stddev: The standard deviation of the mean.
        mask: Boolean array with the same length as y. Values corresponding to
                outliers in y are False. All other values are True.
    """
    # First, make a robust estimate of the standard deviation of y, assuming y is
    # normally distributed. The conversion factor of 1.4826 takes the median
    # absolute deviation to the standard deviation of a normal distribution.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    absdev = np.abs(y - np.median(y))
    sigma = 1.4826 * np.median(absdev)

    # If the previous estimate of the standard deviation using the median absolute
    # deviation is zero, fall back to a robust estimate using the mean absolute
    # deviation. This estimator has a different conversion factor of 1.253.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    if sigma < 1.0e-24:
        sigma = 1.253 * np.mean(absdev)

    # Identify outliers using our estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # of non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers. The
    # following formula is an approximation, see
    # http://w.astro.berkeley.edu/~johnjohn/idlprocs/robust_mean.pro.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Identify outliers using our second estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # with non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Final estimate is the sample mean with outliers removed.
    mean = np.mean(y[mask])
    mean_stddev = sigma / np.sqrt(len(y) - 1.0)

    return mean, mean_stddev, mask


def fit_kepler_spline(all_time,
                                            all_flux,
                                            bkspace_min=0.5,
                                            #bkspace_max=20,
                                            bkspace_max=15,
                                            bkspace_num=20,
                                            maxiter=5,
                                            penalty_coeff=1.0,
                                            verbose=True,
                                            QUICK_SPLINE=False):
    """Fits a Kepler spline with logarithmically-sampled breakpoint spacings.

    Args:
        all_time: List of 1D numpy arrays; the time values of the light curve.
        all_flux: List of 1D numpy arrays; the flux values of the light curve.
        bkspace_min: Minimum breakpoint spacing to try.
        bkspace_max: Maximum breakpoint spacing to try.
        bkspace_num: Number of breakpoint spacings to try.
        maxiter: Maximum number of attempts to fit each spline after removing badly
                fit points.
        penalty_coeff: Coefficient of the penalty term for using more parameters in
                the Bayesian Information Criterion. Decreasing this value will allow
                more parameters to be used (i.e. smaller break-point spacing), and
                vice-versa.
        verbose: Whether to log individual spline errors. Note that if bkspaces
                contains many values (particularly small ones) then this may cause
                logging pollution if calling this function for many light curves.

    Returns:
        spline: List of numpy arrays; values of the best-fit spline corresponding to
                to the input flux arrays.
        metadata: Object containing metadata about the spline fit.
    """

    ## account for when default bkspace_max > length of total lightcurve
    #time_span = all_time[0][-1] - all_time[0][0]
    #if time_span < bkspace_max:
    #    bkspace_max = time_span

    # Logarithmically sample bkspace_num candidate break point spacings between
    # bkspace_min and bkspace_max.
    bkspaces = np.logspace(
            np.log10(bkspace_min), np.log10(bkspace_max), num=bkspace_num)

    return choose_kepler_spline(
        all_time,
        all_flux,
        bkspaces,
        maxiter=maxiter,
        penalty_coeff=penalty_coeff,
        verbose=verbose, QUICK_SPLINE=QUICK_SPLINE)


#######################~~~~~~~~ FINAL PROCESSING STUFF: ~~~~~~~###########################



def MedianBin(time,flux,centroid,normals,t_min,t_max,nbins,viewtype='glob',ndurs=5):
    #Performing median bin
    # - time
    # - flux (normalised to 0)
    # - centroid (normalised to 0)
    # - normals = depth of minimum point and RMS of lcurve relative to deepest depth
    # - t_min
    # - t_max
    # - nbins
    # - viewtype (local or global view)
    # - ndurs (number of durations for local view window)
    
    bins=np.linspace(t_min,t_max,nbins+1)
    bins_i=np.digitize(time,bins)
    view=np.array([np.nanmedian(flux[bins_i==i]) for i in np.arange(1,nbins+1)])
    
    centroid=centroid[~np.isnan(centroid)]
    bins_i_cent=np.digitize(time[~np.isnan(centroid)],bins)
    view_cen=np.array([np.nanmedian(centroid[bins_i_cent==i]) for i in np.arange(1,nbins+1)])
    
    if normals is not None:
        #normalising, (centroids done depending on type as not-transit section required)
        view/=normals[0]
        if viewtype=='loc':
            view_cen*=normals[1]/transitfree_std(view_cen,'loc',ndurs)
        elif viewtype=='glob':
            view_cen*=normals[1]/transitfree_std(view_cen,'glob')
    
    return np.column_stack((np.array(view),np.array(view_cen)))

def transitfree_std(y,viewtype='glob',ndurs=5):
    if viewtype=='glob':
        #Taking std from points from 6.6% to 40% and from 60% to 93.3%
        oot=np.hstack((y[int(np.floor(len(y)*0.066)):int(np.ceil(len(y)*0.4))],
                                    y[int(np.floor(len(y)*0.5)):int(np.ceil(len(y)*0.933))]))
        return np.nanstd(oot)
    elif viewtype=='loc':
        #Taking std from points up to -0.666 duration and from 0.666 dur to the end
        oot=np.hstack((y[0:int(np.ceil((0.5*len(y))*((ndurs-1.33)/ndurs)))],
                                    y[int(np.floor((0.5*len(y))*((ndurs+1.33)/ndurs))):]))
        return np.nanstd(oot)

#######################~~~~~~~~ ACTUAL PROCESSING: ~~~~~~~###########################


def Process(tess_data_dir, tce_row, simrun='TSOP-301-4-sector-run', out_data_dir='Processed', nglob=1001,nloc=101,ndurs=5,cut_anoms=True,dospline=False,use_dv=False):
    # Get filenames
    fnames = tess_filenames(tess_data_dir, tce_row['kepid'], sector=tce_row['sector'], simrun='TSOP-301-4-sector-run',use_dv=use_dv)
    # Get filenames

    if (tce_row['tce_duration']/tce_row['tce_period']**0.33333)>0.25:
        #Given stellar dens from 0.5 to 15 solar density and b from 0.0 to 0.87, 0.02 < dur/P^0.333 < 0.17, so anything over 0.17 is the result of the wrong duration
        dur=tce_row['tce_duration']/24.0
    else:
        dur=tce_row['tce_duration']

    all_time, all_flux, all_centroid, valid_indices = read_tess_light_curve(fnames, use_dv=use_dv, tce=tce_row)
    #spline
    if dospline:
        #print('post-read.  flux',all_flux,'cent', all_centroid)
        time, flux, centroid=process_light_curve(all_time, all_flux, all_centroid, QUICK_SPLINE=True,only_cent=use_dv)
        centroid=-1*centroid
        #print('post-spline.  flux',flux,'cent',centroid)
    else:
        time, flux = all_time, all_flux
        all_centroid=np.array(all_centroid)
        #Getting "radius"
        centroid = -1*np.sqrt(all_centroid.T[0]**2 + all_centroid.T[1]**2 )

    #Normalising flux and centroid to zero:
    flux/=np.nanmedian(flux)
    flux-=np.nanmedian(flux)
    centroid-=np.nanmedian(centroid)
    #print('post-norm. flux',flux,'cent',centroid)

    #If number of durations the "local" view covers is more than a full period, setting to one period
    if ndurs*dur>tce_row['tce_period']:
        ndurs=tce_row['tce_period']/dur

    sector=tce_row['sector']
    if sector is None or sector == 99 or sector=='s1-s4' or sector=='allsectors-99':
        sector=99
    elif type(sector)==str and '-' in sector:
        sector=int(sector.split('-')[1])
    else:
        sector=int(sector)

    out_fname=str(int(tce_row['kepid'])).zfill(12)+'_'+str(int(tce_row['tce_plnt_num'])).zfill(2)+'_'+str(int(sector)).zfill(2)

    #folding
    fold_time, fold_flux, fold_centroid = phase_fold_and_sort_light_curve(time, flux, centroid, tce_row['tce_period'], tce_row['tce_time0bk'],cut_anoms=cut_anoms)
    
    #print('post-fold. flux',fold_flux,'cent',fold_centroid)

    #Doing local view binning, and using this to get normalisation constant:
    locs=MedianBin(fold_time,fold_flux,fold_centroid,None,
                   -0.5*ndurs*dur,
                   0.5*ndurs*dur,
                   nloc,'loc')
    
    #print('post-bin. flux',locs[:,0],'cent',locs[:,1])
    
    depth_norm=abs(np.nanmedian(locs[int(np.ceil((0.5*nloc)*((ndurs-0.33)/ndurs))):int(np.floor((0.5*nloc)*((ndurs+0.33)/ndurs))),0]))

    #Doing global view binning
    globs=MedianBin(fold_time,fold_flux,fold_centroid,None,
                    -0.5*tce_row['tce_period'],0.5*tce_row['tce_period'],
                    nglob,'glob')
    #Normalising to average depth
    locs[:,0]/=depth_norm
    globs[:,0]/=depth_norm

    #Finding RMs of (normalised) global lightcurve, to act as scaling vector for centroids
    rms_norm=transitfree_std(globs[0])
    locs[:,1]*=rms_norm/transitfree_std(locs[:,1],'loc',ndurs)
    globs[:,1]*=rms_norm/transitfree_std(globs[:,1])

    np.save(path.join(out_data_dir,out_fname+'_loc.npy'),locs)
    np.save(path.join(out_data_dir,out_fname+'_glob.npy'),globs)

    fold_time[fold_time<0]+=tce_row['tce_period']
    #As data is -0.5Per -> 0.5Per, need to shift time to
    locs_sec=MedianBin(np.where(fold_time<0,fold_time+tce_row['tce_period'],fold_time),
                       fold_flux,fold_centroid,[depth_norm,rms_norm],
                       0.5*tce_row['tce_period']-0.5*ndurs*dur,
                       0.5*tce_row['tce_period']+0.5*ndurs*dur,
                       nloc,'loc',ndurs)

    np.save(path.join(out_data_dir,out_fname+'_sec.npy'),locs_sec)

    fold_time_p2, fold_flux_p2, fold_centroid_p2 = phase_fold_and_sort_light_curve(time, flux, centroid, tce_row['tce_period']*2, tce_row['tce_time0bk'],cut_anoms=cut_anoms)

    p2locs_odd=MedianBin(fold_time_p2,fold_flux_p2,fold_centroid_p2,[depth_norm,rms_norm],
                         -0.5*ndurs*dur,
                         0.5*ndurs*dur,
                         nloc,'loc',ndurs)
    p2locs_even=MedianBin(np.where(fold_time_p2<0,fold_time_p2+2*tce_row['tce_period'],fold_time_p2),
                          fold_flux_p2,fold_centroid_p2,[depth_norm,rms_norm],
                          tce_row['tce_period']-0.5*ndurs*dur,
                          tce_row['tce_period']+0.5*ndurs*dur,
                          nloc,'loc',ndurs)
    bothp2s=np.column_stack((p2locs_odd,p2locs_even))
    np.save(path.join(out_data_dir,out_fname+'_odd_even.npy'),bothp2s)

    info=tce_row.values.ravel()
    np.save(path.join(out_data_dir,out_fname+'_info.npy'),info)

    return [globs,locs,locs_sec,bothp2s,info,np.column_stack((np.array(time).ravel(),
                                                              np.array(flux).ravel(),
                                                              np.array(centroid).ravel()))]

#######################~~~~~~~~ RUNNING: ~~~~~~~###########################

if __name__=="__main__":
    #
    # python QuickProcess.py csvfile.csv label_eg_train /data/directory/to/save
    #
    tcecsv=pd.DataFrame.from_csv(sys.argv[1])
    label=sys.argv[2]
    datadir=sys.argv[3]
    nglob=sys.argv[4]
    nloc=sys.argv[5]
    for n in range(len(tcecsv)):
        save_loc=path.join(datadir,label)
        if not path.exists(datadir):
            os.system('mkdir '+datadir)
        if not path.exists(save_loc):
            os.system('mkdir '+save_loc)
        _ = Process('/home/hosborn/TESS/fdldata/lilithSimulatedData/', tcecsv.iloc[n], simrun='TSOP-301-4-sector-run',
                    out_data_dir=save_loc,nglob=int(nglob),nloc=int(nloc),ndurs=5,cut_anoms=True,dospline=True,use_dv=True)
        print(tcecsv.index.values[n],tcecsv.kepid.values[n])
