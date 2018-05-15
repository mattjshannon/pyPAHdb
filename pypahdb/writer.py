#!/usr/bin/env python
# writer.py

"""
writer.py: Writes decomposer results to file

This file is part of pypahdb - see the module docs for more
information.
"""

import copy
import decimal
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import time

from .decomposer import decomposer
from astropy.io import fits
# from astropy import wcs
from matplotlib.backends.backend_pdf import PdfPages


class writer(object):
    """Creates a writer object.

    Writes PDF and FITS files.

    Attributes:
        result (pypahdb.decomposer): Decomposer object.
        header (fits.header.Header): FITS header (normally).
        basename (str): Base string, used for saving files.
        opdf (bool): Whether to save a summary PDF or not. True by default.
        ofits (bool): Whether to save a FITS file for the fit. True by default.

    """

    def __init__(self, result, header="", basename="", opdf=True, ofits=True):
        """Instantiate a writer object.

        Args:
            result (pypahdb.decomposer): Decomposer object.

        Keywords:
            header (String, list): header
            basename (String): Prefix for filenames to be written.
            opdf (bool, optional): Whether to save a summary PDF or not.
                True by default.
            ofits (bool, optional): Whether to save a FITS file for the fit.
                True by default.

        """
        self.result = result
        self.header = header
        self.basename = basename
        self.opdf = opdf
        self.ofits = ofits

        # What if not decomposer object ...
        # Make sure we're dealing with a 'decomposer' object
        if isinstance(result, decomposer):
            if opdf:
                # Save a PDF with model fit information.
                self._save_summary_pdf()

            if ofits:
                # Save fit results to a FITS file.
                if isinstance(header, fits.header.Header):
                    # should probably clean up the header, i.e.,
                    # extract certain keywords only
                    hdr = copy.deepcopy(header)
                else:
                    hdr = fits.Header()

                self._save_fits(hdr)

    def _plot_pahdb_fit(self, i, j):
        """Plot a pyPAHdb fit and save to a PDF.

        Note:
            Designed to accept (i,j) to accommodate spectral cubes.

        Args:
            i (int): Pixel coordinate (abscissa).
            j (int): Pixel coordinate (ordinate).

        Returns:
            fig (mpl.figure): Figure object from matplotlib.

        """

        def smart_round(value, style="0.1"):
            """Round a float correctly, returning a string."""
            tmp = decimal.Decimal(value).quantize(decimal.Decimal(style))
            return str(tmp)

        # Create figure, shared axes.
        fig = plt.figure(figsize=(8, 11))
        gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 2, 2])
        gs.update(wspace=0.025, hspace=0.00)  # set the spacing between axes.
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax2 = fig.add_subplot(gs[2], sharex=ax0)
        ax3 = fig.add_subplot(gs[3], sharex=ax0)

        # Common abscissa.
        wave = 1e4 / self.result.spectrum.abscissa

        # ######## AX0 ######## #
        # Plot data and overall model fit.
        data = self.result.spectrum.ordinate[:, i, j]
        model = self.result.fit[:, i, j]
        ax0.plot(wave, data, 'x', ms=5, mew=0.5, label='input', color='black')
        ax0.plot(wave, model, label='fit', color='red')

        # Overplot the "norm" as text.
        norm_val = self.result.norm[i][j]
        norm_str = smart_round(norm_val, style="0.1")
        ax0.text(0.025, 0.9, '$norm$=' + norm_str, ha='left', va='center',
                 transform=ax0.transAxes)

        # ######## AX1 ######## #
        # Plot the residuals of ( data - model ).
        ax1.plot(wave, data - model, lw=1, label='residual', color='black')
        ax1.axhline(y=0, color='0.5', ls='--', dashes=(12, 16), zorder=-10,
                    lw=0.5)

        # ######## AX2 ######## #
        # Model breakdown: by size.
        ax2.plot(wave, model, color='red', lw=1.5)
        y_small = self.result.size['small'][:, i, j]
        y_large = self.result.size['large'][:, i, j]
        ax2.fill_between(wave, y_large * 0, y_large, label='large',
                         color='xkcd:blue', alpha=0.3, zorder=2)
        ax2.fill_between(wave, y_small * 0, y_small, label='small',
                         color='xkcd:green', alpha=0.4, zorder=1)
        ax2.plot(wave, self.result.size['large'][:, i, j],
                 lw=1, color='xkcd:blue', zorder=2)
        ax2.plot(wave, self.result.size['small'][:, i, j],
                 lw=1, color='xkcd:green', alpha=0.4, zorder=1)

        # Overplot the large fraction as text.
        size_frac = self.result.large_fraction[i][j]
        size_str = smart_round(size_frac, "0.01")
        size_str = '$f_{large}$=' + size_str
        ax2.text(0.025, 0.9, size_str, ha='left', va='center',
                 transform=ax2.transAxes)

        # ######## AX3 ######## #
        # Model breakdown: by charge.
        ax3.plot(wave, model, color='red', lw=1.5)
        ax3.plot(wave, self.result.charge['anion'][:, i, j],
                 label='anion', lw=1, color='orange')
        ax3.plot(wave, self.result.charge['neutral'][:, i, j],
                 label='neutral', lw=1, color='green')
        ax3.plot(wave, self.result.charge['cation'][:, i, j],
                 label='cation', lw=1, color='blue')

        # Overplot the ionized fraction as text.
        ion_frac = self.result.ionized_fraction[i][j]
        ion_str = smart_round(ion_frac, "0.01")
        cat_str = '$f_{ionized}$=' + ion_str
        ax3.text(0.025, 0.9, cat_str, ha='left', va='center',
                 transform=ax3.transAxes)

        # Plot axes/figure labels.
        title_str = 'Cube: (i, j) = ' + '(' + str(i) + ', ' + str(j) + ')'
        ax0.set_title(title_str)
        ylabel = self.result.spectrum.units['ordinate']['str']
        if ylabel == 'surface brightness [MJy/sr]':
            ylabel = 'Surface Brightness [MJy/sr]'
        xlabel = self.result.spectrum.units['abscissa']['str']
        if xlabel == 'wavelength [micron]':
            xlabel = 'Wavelength [μm]'
        fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical')
        ax3.set_xlabel(xlabel)

        # Set tick parameters and add legends to all axes.
        for ax in (ax0, ax1, ax2, ax3):
            ax.tick_params(axis='both', which='both', direction='in',
                           top=True, right=True)
            ax.minorticks_on()
            ax.legend(loc=0, frameon=False)

        return fig

    def _save_summary_pdf(self):
        """Save a PDF summarizing the goodness of fit (plots, few #s).

        Note:
            Iterates over a spectral cube (with i, j pixels) and creates one
            page for each fit, all of which are stitched into a single PDF
            as output. Utilizes self.basename for determining output filename.

        Returns:
            If successful.

        """

        with PdfPages(self.basename + 'pypahdb.pdf') as pdf:
            d = pdf.infodict()
            d['Title'] = 'pyPAHdb Result Summary'
            d['Author'] = 'pyPAHdb'
            d['Subject'] = 'Summary of a pyPAHdb PAH database Decomposition'
            d['Keywords'] = 'pyPAHdb PAH database'
            for i in range(self.result.spectrum.ordinate.shape[1]):
                for j in range(self.result.spectrum.ordinate.shape[2]):
                    fig = self._plot_pahdb_fit(i, j)
                    pdf.savefig(fig)
                    plt.close(fig)
                    plt.gcf().clear()
            # fig = self._plot_pahdb_fit(7, 5)
            # pdf.savefig(fig)
            # plt.close(fig)
            # plt.gcf().clear()
        return

    def _save_fits(self, hdr):
        """Save a FITS file with the model fit details.

        Returns:
            If successful.

        """

        hdr['DATE'] = time.strftime("%Y-%m-%dT%H:%m:%S")
        hdr['SOFTWARE'] = "pypahdb"
        hdr['SOFT_VER'] = "0.5.0.a1"
        hdr['COMMENT'] = "This file contains the results from a pypahdb fit"
        hdr['COMMENT'] = "Visit https://github.com/pahdb/pypahdb/ for more " \
                         "information on pypahdb"
        hdr['COMMENT'] = "The 1st plane contains the ionized fraction"
        hdr['COMMENT'] = "The 2nd plane contains the large fraction"
        hdr['COMMENT'] = "The 3rd plane contains the norm"

        # write results to fits-file
        hdu = fits.PrimaryHDU(np.stack((self.result.ionized_fraction,
                                        self.result.large_fraction,
                                        self.result.norm), axis=0), header=hdr)
        hdu.writeto(self.basename + 'pypahdb.fits', overwrite=True)

        return
