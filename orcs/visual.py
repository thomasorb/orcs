import pylab as pl
from matplotlib.widgets import Slider, Button
from matplotlib import gridspec
import numpy as np
from catalog import Catalog
from process import SpectralCube

class Viewer(object):

    def __init__(self, cube, catalog_path=None):

        if isinstance(cube, str):
            self.cube = SpectralCube(cube)
        elif isinstance(cube, SpectralCube):
            self.cube = cube
        else: raise TypeError('cube can be a path to a cube or an instance of a orcs.core.SpectralCube')
        self.fig = pl.figure(figsize=(10, 7))
        self.imshown = None
        gs  = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 0.05])
        ax0 = pl.subplot(gs[0])
        ax1 = pl.subplot(gs[1])
        ax2 = pl.subplot(gs[2])
        ax3 = pl.subplot(gs[4])

        self.axes = (ax0, ax1, ax2, ax3)
        self.xlim = None
        self.ylim = None
        self.show_df()
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.radius = 10
        self.sradius = Slider(self.axes[2], 'Radius', 1, 30.0, valinit=self.radius)
        self.sradius.on_changed(self.radius_update)
        self.bnorm = Button(self.axes[3], 'Normalize Colorbar')
        self.bnorm.on_clicked(self.norm)

        if catalog_path is not None:
            self.cat = Catalog(catalog_path)
        else:
            self.cat = None

        pl.show()

    def show_df(self, vmin=None, vmax=None):
        df = self.cube.get_deep_frame().T
        if vmin is None and vmax is None and self.imshown is not None:
            vmin, vmax = self.imshown.get_clim()
        self.imshown = self.axes[0].imshow(
            df, origin='bottom-left', vmin=vmin, vmax=vmax)
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.button == 3 and event.inaxes == self.axes[0]:
            self.plot_spectrum(event.xdata, event.ydata, self.radius)

    def plot_spectrum(self, x, y, r):
        # get data
        axis, spec = self.cube.extract_spectrum(
            x, y, r)
        self.axes[1].cla()
        self.axes[1].plot(axis, spec, label='{:.1f} {:.1f}'.format(
            x, y))
        self.axes[1].grid()
        self.axes[1].set_xlabel(r'Wavenumber (cm$^{-1}$)')
        self.axes[1].set_ylabel(r'Flux (erg/cm$^{2}$/s/$\AA$)')
        self.axes[1].legend()
        self.xlim = self.axes[0].get_xlim()
        self.ylim = self.axes[0].get_ylim()
        self.axes[0].cla()

        self.axes[0].set_xlim(self.xlim)
        self.axes[0].set_ylim(self.ylim)
        self.axes[0].axvline(x=x, c='orange', alpha=0.7)
        self.axes[0].axhline(y=y, c='orange', alpha=0.7)
        circle = pl.Circle((x, y), r,
                           color='orange', fill=False, alpha=0.7)
        self.axes[0].add_artist(circle)
        self.show_df()
        self.fig.canvas.draw()

    def radius_update(self, val):
        self.radius = float(val)

    def norm(self, _):
        self.xlim = self.axes[0].get_xlim()
        self.ylim = self.axes[0].get_ylim()
        df = self.cube.get_deep_frame().T
        box = df[int(self.xlim[0]):int(self.xlim[1]),
                 int(self.ylim[0]):int(self.ylim[1])]
        vmin = np.nanpercentile(box, 1)
        vmax = np.nanpercentile(box, 99)
        self.show_df(vmin=vmin, vmax=vmax)


class Checker():


    def __init__(self, inov, x, y, cat, cube, det_frame, detb, nov_add, cm1_axis, outfile, candfile,
                 hdr, dxmap, dymap, boxsize=51):
        self.contrast = 100
        self.boxsize = boxsize + int(~boxsize&1) # always odd
        self.outfile = outfile
        self.candfile = candfile
        self.cm1_axis = cm1_axis
        self.cat = cat
        self.x = x
        self.y = y
        self.x_orig = int(x)
        self.y_orig = int(y)
        self.cube = cube
        self.inov = inov
        self.nov_add = nov_add
        self.detb = detb
        self.det_frame = det_frame
        self.hdr = hdr
        self.dxmap = dxmap
        self.dymap = dymap

        self.add_detb = dict()
        self.add_detb['xmatch'] = 'None'
        self.add_detb['xmatch_r'] = 'None'
        self.add_detb['x'] = 'None'
        self.add_detb['y'] = 'None'


        self.is_show = False

        self.show()

    def show(self):

        xminb, xmaxb, yminb, ymaxb = orb.utils.image.get_box_coords(
            int(self.x), int(self.y), self.boxsize,
            0, self.det_frame.shape[0],
            0, self.det_frame.shape[1])

        self.xminb = xminb
        self.yminb = yminb

        bbox = self.det_frame[xminb:xmaxb, yminb:ymaxb]
        self.xc, self.yc = self.x_orig-xminb, self.y_orig-yminb

        if not self.is_show:
            self.fig, (self.ax, self.ax2) = pl.subplots(1,2, figsize=(16,8))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        else:
            self.ax.cla()

        self.ax.imshow(bbox.T,
                       vmin=np.nanpercentile(bbox, 2),
                       vmax=np.nanpercentile(bbox, self.contrast),
                       cmap='gray',
                       origin='bottom-left')

        self.src = dict()
        __xpos = list()
        __ypos = list()
        __id = list()
        __r = list()

        for ixy in range(len(self.cat['xpos'])):
            __x = self.cat['xpos'][ixy] - xminb
            __y = self.cat['ypos'][ixy] - yminb
            if (__x < self.boxsize) and (__x > 0) and (__y < self.boxsize) and (__y > 0):
                __xpos.append(__x)
                __ypos.append(__y)
                print self.cat['index'][ixy], __x, __y, ixy
                __id.append(self.cat['index'][ixy])
                __r.append(np.sqrt((__x - self.xc)**2. + (__y - self.yc)**2.))

        self.src['xpos'] = __xpos
        self.src['ypos'] = __ypos
        self.src['id'] = __id
        self.src['r'] = __r
        self.src['nb'] = len(__xpos)

        if self.nov_add is not None:
            err_rad = max(self.nov_add[0][self.inov], self.nov_add[1][self.inov]) / 0.32
        else:
            err_rad = 1

        self.ax.add_artist(pl.Circle((self.xc, self.yc), err_rad, color='blue',  fill=False))
        if self.nov_add is not None:
            id_ = self.nov_add[2][self.inov]
        else:
            id_ = ''
        [[self.ra, self.dec]] = orb.utils.astrometry.pix2world(
            self.hdr, self.det_frame.shape[0], self.det_frame.shape[1],
            np.array([[float(self.x),float(self.y)]]), self.dxmap, self.dymap)

        self.ra = orb.utils.astrometry.deg2ra(self.ra)
        self.dec = orb.utils.astrometry.deg2dec(self.dec)

        self.ax.set_title('{} ({:.1f}, {:.1f}) ({:.0f}:{:.0f}:{:.2f}, {:.0f}:{:.0f}:{:.2f})'.format(
            id_, self.x, self.y, self.ra[0], self.ra[1], self.ra[2], self.dec[0], self.dec[1], self.dec[2]))


        self.ax.scatter(self.src['xpos'], self.src['ypos'])
        for icat in range(self.src['nb']):
            self.ax.text(self.src['xpos'][icat] + 2,
                         self.src['ypos'][icat] + 2,
                         self.src['id'][icat],
                         color='red', fontsize='14')

        self.ax.axvline(self.boxsize/2, color='red', alpha=0.5)
        self.ax.axhline(self.boxsize/2, color='red', alpha=0.5)


        if not self.is_show:
            self.is_show = True
            pl.show()
        else:
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'x':
            xmatch = None
            min_r = 1e18
            for isrc in range(self.src['nb']):
                if self.src['r'][isrc] < min_r:
                    xmatch = self.src['id'][isrc]
                    min_r = self.src['r'][isrc]
            if xmatch is not None:
                self.add_detb['xmatch'] = int(xmatch)
                self.add_detb['xmatch_r'] = min_r

        if event.key == 'y':
            self.add_detb['x'] = self.x
            self.add_detb['y'] = self.y
            self.save()

        if event.key == 'c':
            with open(self.candfile, 'a') as f:
                f.write('{} {}\n'.format(self.x, self.y))


        if event.key == 'q':
            self.write()
            quit()

        if event.key == '-':
            self.contrast -= 0.2
            self.show()

        if event.key == '+':
            self.contrast += 0.2
            if self.contrast > 100: self.contrast = 100
            self.show()

        if event.key == 'o':
            self.x = self.x_orig
            self.y = self.y_orig
            self.plot_spec()
            self.show()

        if event.key == 'left':
            self.x -= 1
            self.plot_spec()
            self.show()

        if event.key == 'right':
            self.x += 1
            self.plot_spec()
            self.show()

        if event.key == 'up':
            self.y += 1
            self.plot_spec()
            self.show()

        if event.key == 'down':
            self.y -= 1
            self.plot_spec()
            self.show()


        if event.key == 'escape':
            self.save()
            pl.close()

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.x = event.xdata + self.xminb
            self.y = event.ydata + self.yminb
        print self.x, self.y

        self.plot_spec()
        self.show()

    def plot_spec(self):
        SMA = 5
        BACK = 25

        xmins, xmaxs, ymins, ymaxs = orb.utils.image.get_box_coords(
            int(self.x), int(self.y), SMA,
            0, self.det_frame.shape[0],
            0, self.det_frame.shape[1])

        lilbox = self.cube[xmins:xmaxs, ymins:ymaxs, :]
        spec = np.nansum(np.nansum(lilbox, axis=0), axis=0)

        xminb, xmaxb, yminb, ymaxb = orb.utils.image.get_box_coords(
            int(self.x), int(self.y), BACK,
            0, self.det_frame.shape[0],
            0, self.det_frame.shape[1])
        bigbox = self.cube[xminb:xmaxb, yminb:ymaxb, :]
        bigbox[xmins - xminb:xmaxs-xminb, ymins - yminb:ymaxs - yminb] = np.nan

        sky = np.nanmedian(np.nanmedian(bigbox, axis=0), axis=0) * SMA**2

        spec -= sky

        self.ax2.cla()
        #self.ax2.axvline(1e7/656.28, color='red')
        self.ax2.axvline(1e7/abs(656.28 * (500/3e5 - 1)), color='red')
        self.ax2.axvline(1e7/abs(656.28 * (-500/3e5 - 1)), color='red')

        #self.ax2.axvline(1e7/654.8, color='green')
        self.ax2.axvline(1e7/abs(654.8 * (500/3e5 - 1)), color='green')
        self.ax2.axvline(1e7/abs(654.8 * (-500/3e5 - 1)), color='green')

        #self.ax2.axvline(1e7/658.4, color='orange')
        self.ax2.axvline(1e7/abs(658.4 * (500/3e5 - 1)), color='orange')
        self.ax2.axvline(1e7/abs(658.4 * (-500/3e5 - 1)), color='orange')

        #self.ax2.plot(self.cm1_axis, sky - np.nanmedian(sky), color='0.5')
        self.ax2.plot(self.cm1_axis, spec, color='0.', lw=2.)
        spec_sm = orb.utils.vector.smooth(spec, 7)
        self.ax2.plot(self.cm1_axis, spec_sm, color='red', lw=2.)


        self.ax2.set_xlim((14600,15400))
        self.ax2.set_ylim((np.nanmin(spec), np.nanmax(spec)))


    def save(self):
        if self.nov_add is not None:
            _id = self.nov_add[2][self.inov]
        else:
            _id = str(self.inov)
        self.detb[_id] = self.add_detb

    def write(self):
        with open(self.outfile, 'a') as outf:
            keys = self.detb[self.detb.keys()[0]]
            for ikey in self.detb.keys():
                _str = ikey + '\t'
                for iikey in keys:
                    _str += '{}\t'.format(self.detb[ikey][iikey])
                print _str
                outf.write(_str + '\n')


    def __del__(self):
        self.save()
