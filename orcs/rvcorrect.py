#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: rvcorrect.py

## Copyright (c) 2010-2015 Thomas Martin <thomas.martin.1@ulaval.ca>
## 
## This file is part of ORCS
##
## ORCS is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ORCS is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORCS.  If not, see <http://www.gnu.org/licenses/>.

import math
import numpy as np

"""Compute the radial velocity correction for spectral data.

The whole class has been created using David Nidever IDL function
RVCORRECT itself translated from the IRAF function RVCORRECT.

The IDL code can be found at::
    
  http://www.astro.virginia.edu/~dln5q/research/idl/rvcorrect.pro

The description of the IRAF function can be found at::
    
  http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?rvcorrect
"""
#################################################
#### RVCORRECT ##################################
#################################################

class RVCorrect():
    """
    Compute radial velocity correction. The whole class has been
    created using David Nidever's IDL function RVCORRECT itself
    translated from the IRAF function RVCORRECT.

    The IDL code can be found at::
    
      http://www.astro.virginia.edu/~dln5q/research/idl/rvcorrect.pro

    The description of the IRAF function can be found at::
    
      http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?rvcorrect
    """

    J2000  = 2000.0 
    JD2000 = 2451545.0
    JYEAR  = 365.25
    RADEG = 180./math.pi

    ra = None # RA of the object
    dec = None # DEC of the object
    year = None # Year of observation (4-digit)
    month = None # Month of observation
    day = None # Day of observation
    ut = None # UT hour of observation
    longitude = None # Observatory longitude
    latitude = None # Observatory latitude
    altitude = None # Observatory altitude
    ep = None # Epoch of the coordinates

    silent = None # If False print all the output parameters 

    hjd = None # HJD : The heliocentric Julian day
    vhelio = None # VHELIO : The heliocentric radial velocity correction
    vlsr = None # VLSR : The LSR radial velocity correction
    vrot = None # VROT : The diurnal rotation velocity
    vbary = None # VBARY : The lunary velocity
    vorb = None # VORB : The annual velocity
    vsol = None # VSOL : The solar velocity

    def __init__(self, ra, dec, date, ut, obs_coords, ep=2000.,
                 silent=False):
        """
        :param ra: RA of the object. The format can be either
          'hours:minutes:seconds' or (hours,minutes,seconds) or a
          decimal hour.
          
        :param dec: DEC of the object. The format can be either
          'deg:minutes:seconds' or (deg, minutes, seconds) or an angle
          in degree.
          
        :param date: Date of the observation (year, month, day).
        
        :param ut: Universal time of the observation. The format can
          be either 'hours:minutes:seconds' or (hours,minutes,seconds)
          or a decimal hour.
          
        :param obs_coords: The observatory coordinates (obs_lat, obs_long,
          obs_alt).
        
        :param ep: (Optional) The epoch of the coordinates (2000. by
          default).

        :param silent: (Optional) If False print all the output
          parameters (default False).
        """
        # RA DEC UT conversion
        if isinstance(ra, basestring):
            ra = ra.split(":")
        if isinstance(dec, basestring):
            dec = dec.split(":")
        if isinstance(ut, basestring):
            ut = ut.split(":")

        if len(ra) == 3:
            ra_conv = abs(float(ra[0])) + float(ra[1])/60. + float(ra[2])/3600.
            if ra[0].find('-') != -1: self.ra = -ra_conv
            else: self.ra = ra_conv
        if len(dec) == 3:
            dec_conv = (abs(float(dec[0])) + float(dec[1])/60.
                        + float(dec[2])/3600.)
            if dec[0].find('-') != -1: self.dec = -dec_conv
            else: self.dec = dec_conv
        if len(ut) == 3:
            ut_conv = abs(float(ut[0])) + float(ut[1])/60. + float(ut[2])/3600.
            if ut[0].find('-') != -1: self.ut = -ut_conv
            else: self.ut = ut_conv

        (self.year, self.month, self.day) = date
        (self.latitude, self.longitude, self.altitude) = obs_coords
        self.ep = ep
        self.silent = silent

    def ast_date_to_julday(self, year, month, day, t):
        """
        Convert date to Julian day. This assumes dates after year 99.

        :param year: Year
        :param month: Month
        :param day: Day
        :param t: Time for date
        """
        if year < 100.:
            y = 1900.0 + year
        else:
            y = year

        if month > 2:
            m = month + 1
        else:
            m = month + 13
            y = y - 1

        jd = int(self.JYEAR * y) + int(30.6001 * m) + day + 1720995.0
        
        if ((day + 31.0 * (m + 12. * y)) >= 588829.0):
            d = int(y / 100.)
            m = int(y / 400.)
            jd = jd + 2.0 - d + m

        jd = jd - 0.5 + int(t * 360000.0 + 0.5) / 360000.0 / 24.0
        return jd

    def ast_date_to_epoch(self, year, month, day, ut):
        """
        Convert Gregorian date and solar mean time to a Julian epoch.
        A Julian epoch has 365.25 days per year and 24 hours per day.

        :param year: Year
        :param month: Month (1-12)
        :param day: Day of month
        :param ut: Universal time for date (mean solar day)
        """

        jd = self.ast_date_to_julday(year, month, day, ut)
        epoch = self.J2000 + (jd - self.JD2000) / self.JYEAR
        return epoch

    def ast_epoch_to_date(self, epoch):
        """
        Convert a Julian epoch to year, month, day, and time.

        :param epoch: Julian epoch
        """
        jd = self.JD2000 + (epoch - self.J2000) * self.JYEAR
        return self.ast_julday_to_date(jd)

    def ast_julday(self, epoch):
        """
        Convert epoch to Julian day.
        
        :param epoch: Epoch
        """
        jd = self.JD2000 + (epoch - self.J2000) * self.JYEAR
        return jd

    def ast_julday_to_date(self, j):
        """
        Convert Julian date to calendar date.  This is taken from
        Numerical Receipes by Press, Flannery, Teukolsy, and
        Vetterling.

        :param j: Julian date
        """
        ja = int(j)
        t = 24. * (j - ja + 0.5)

        if (ja >= 2299161):
            jb = float(int(((ja - 1867216) - 0.25) / 36524.25))
            ja = ja + 1. + jb - int(float(jb) / 4.)

        jb = ja + 1524.
	jc = float(int(6680. + ((jb - 2439870) - 122.1) / self.JYEAR))
	jd = 365. * jc + int(jc / 4.)
	je = float(int(float(jb - jd) / 30.6001))
	day = float(jb - jd - int(30.6001 * je))
	month = float(je - 1)

        if (month >= 12):
            month -= 12.
            
        year = jc - 4715.

        if (month > 2.):
            year -= 1.
        if (year < 0.):
            year -= 1.

        return year, month, day, t


    def ast_mst(self, epoch, longitude):
        """
        Mean sidereal time of the epoch at the given longitude.  This
        procedure may be used to optain Greenwich Mean Sidereal Time
        (GMST) by setting the longitude to 0.

        :param epoch: Epoch
        :param longitude: Longitude in degrees
        """

        # Determine JD and UT, and T (JD in centuries from J2000.0).
        jd = self.ast_julday(epoch)
	ut = (jd - int(jd) - 0.5) * 24.0
	t = (jd - 2451545.) / 36525.
        # The GMST at 0 UT in seconds is a power series in T.
        st = 24110.54841 + t * (8640184.812866 + t * (0.093104 - t * 6.2e-6))
        # Correct for longitude and convert to standard hours.
        st = (st / 3600. + ut - longitude / 15.) % 24.0

        if (st < 0):
            st += 24.

        return st

    def ast_precess(self, ra1, dec1, epoch1, epoch2):
        """
        Precess coordinates from epoch1 to epoch2.
        
        The method used here is based on the new IAU system described
        in the supplement to the 1984 Astronomical Almanac.  The
        precession is done in two steps; precess epoch1 to the
        standard epoch J2000.0 and then precess from the standard
        epoch to epoch2.  The precession between any two dates is done
        this way because the rotation matrix coefficients are given
        relative to the standard epoch.

        :param ra1: RA for first coordinates
        :param dec1: DEC for first coordinates
        :param epoch 1: Epoch for first coordinates
        :param epoch 2: Epoch for second coordinates
        """

        r0 = np.zeros(3, dtype=float)
        r1 = np.zeros(3, dtype=float)
        p = np.zeros((3,3), dtype=float)
        
        # If the input epoch is 0 or undefined then assume the input
        # epoch is the same as the output epoch.  If the two epochs
        # are the same then return the coordinates from epoch1.

        if ((epoch1 == 0.) or (epoch1 == epoch2)):
            ra2 = ra1
            dec2 = dec1
            return ra2, dec2

        # Rectangular equitorial coordinates (direction cosines).
        ra2 = (ra1 * 15.)/self.RADEG
        dec2 = dec1/self.RADEG

        r0[0] = math.cos(ra2) * math.cos(dec2)
	r0[1] = math.sin(ra2) * math.cos(dec2)
	r0[2] = math.sin(dec2)

        # If epoch1 is not the standard epoch then precess to the
	# standard epoch.
        if (epoch1 != 2000.):
            p = self.ast_rotmatrix(epoch1)
            r1[0] = p[0, 0] * r0[0] + p[0, 1] * r0[1] + p[0, 2] * r0[2]
	    r1[1] = p[1, 0] * r0[0] + p[1, 1] * r0[1] + p[1, 2] * r0[2]
	    r1[2] = p[2, 0] * r0[0] + p[2, 1] * r0[1] + p[2, 2] * r0[2]
	    r0[0] = r1[0]
	    r0[1] = r1[1]
	    r0[2] = r1[2]

        # If epoch2 is not the standard epoch then precess to the
	# standard epoch.
        if (epoch2 != 2000.):
            p = self.ast_rotmatrix(epoch2)
            r1[0] = p[0, 0] * r0[0] + p[1, 0] * r0[1] + p[2, 0] * r0[2]
	    r1[1] = p[0, 1] * r0[0] + p[1, 1] * r0[1] + p[2, 1] * r0[2]
	    r1[2] = p[0, 2] * r0[0] + p[1, 2] * r0[1] + p[2, 2] * r0[2]
	    r0[0] = r1[0]
	    r0[1] = r1[1]
	    r0[2] = r1[2]

        # Convert from radians to hours and degrees.
        ra2 = (math.atan2(r0[1], r0[0]) / 15.) * self.RADEG
	dec2 =(math.asin(r0[2])) * self.RADEG
        if (ra2 < 0.):
            ra2 = ra2 + 24.0

        return (ra2, dec2)

    def ast_rotmatrix(self, epoch):
        """
        Compute the precession rotation matrix from the standard epoch
        J2000.0 to the specified epoch.

        :param epoch: Epoch of date
        """
        p = np.zeros((3,3), dtype=float)
        
        # The rotation matrix coefficients are polynomials in time
	# measured in Julian centuries from the standard epoch.  The
	# coefficients are in degrees.

        t = (self.ast_julday(epoch) - 2451545.0) / 36525.

	a = t * (0.6406161 + t * (0.0000839 + t * 0.0000050))
	b = t * (0.6406161 + t * (0.0003041 + t * 0.0000051))
	c = t * (0.5567530 - t * (0.0001185 + t * 0.0000116))

        # Compute the cosines and sines once for efficiency.
        ca = math.cos(a/self.RADEG)
	sa = math.sin(a/self.RADEG)
	cb = math.cos(b/self.RADEG)
	sb = math.sin(b/self.RADEG)
	cc = math.cos(c/self.RADEG)
	sc = math.sin(c/self.RADEG)
        
        # Compute the rotation matrix from the sines and cosines.
        p[0, 0] = ca * cb * cc - sa * sb
	p[1, 0] = -sa * cb * cc - ca * sb
	p[2, 0] = -cb * sc
	p[0, 1] = ca * sb * cc + sa * cb
	p[1, 1] = -sa * sb * cc + ca * cb
	p[2, 1] = -sb * sc
	p[0, 2] = ca * sc
	p[1, 2] = -sa * sc
	p[2, 2] = cc
        
        return p

    def ast_coord(self, ao, bo, ap, bp, a1, b1):
        """Convert spherical coordinates to new system.
        
        This procedure converts the longitude-latitude coordinates
        (a1, b1) of a point on a sphere into corresponding coordinates
        (a2, b2) in a different coordinate system that is specified by
        the coordinates of its origin (ao, bo).  The range of a2 will
        be from -pi to pi.

        :param ao: longitude of the origin of the new coordinates (radians)
        :param bo: latitude of the origin of the new coordinates (radians)
        :param ap: longitude of the pole of the new coordinates (radians)
        :param bp: latitude of the pole of the new coordinates (radians)
        :param a1: longitude of the pole of the coordinates to be
          converted (radians)
        :param b1: latitude of the pole of the coordinates to be
          converted (radians)
        """
        x = math.cos(a1) * math.cos(b1)
	y = math.sin(a1) * math.cos(b1)
	z = math.sin(b1)
	xp = math.cos(ap) * math.cos(bp)
	yp = math.sin(ap) * math.cos(bp)
	zp = math.sin(bp)

        # Rotate the origin about z.
	sao = math.sin(ao)
	cao = math.cos(ao)
	sbo = math.sin(bo)
	cbo = math.cos(bo)
	temp = -xp * sao + yp * cao
	xp = xp * cao + yp * sao
	yp = temp
	temp = -x * sao + y * cao
	x = x * cao + y * sao
	y = temp

        # Rotate the origin about y.
	temp = -xp * sbo + zp * cbo
	xp = xp * cbo + zp * sbo
	zp = temp
	temp = -x * sbo + z * cbo
	x = x * cbo + z * sbo
	z = temp

        # Rotate pole around x.
	sbp = zp
	cbp = yp
	temp = y * cbp + z * sbp
	y = y * sbp - z * cbp
	z = temp

        # Final angular coordinates.
	a2 = math.atan2(y, x)
	b2 = math.asin(z)

        return a2, b2

    def ast_hjd(self, ra, dec, epoch):
        """Heliocentric Julian Day from Epoch

        :param ra: RA of observation (hours)
        :param dec: DEC of observation (degrees)
        :param epoch: Julian epoch of observation
        """
        return self.ast_jd_to_hjd(ra, dec, self.ast_julday(epoch))

    def ast_jd_to_hjd(self, ra, dec, jd):
        """Heliocentric Julian Day from UT Julian date

        :param ra: RA of observation (hours)
        :param dec: DEC of observation (degrees)
        :param jd: Geocentric Julian date of observation
        """
        # JD is the geocentric Julian date.
	# T is the number of Julian centuries since J1900.
	t = (jd - 2415020.) / 36525.

        # MANOM is the mean anomaly of the Earth's orbit (degrees)
	# LPERI is the mean longitude of perihelion (degrees)
	# OBLQ is the mean obliquity of the ecliptic (degrees)
	# ECCEN is the eccentricity of the Earth's orbit (dimensionless)
        manom = 358.47583 + t * (35999.04975 - t * (0.000150 + t * 0.000003))
        lperi = 101.22083 + t * (1.7191733 + t * (0.000453 + t * 0.000003))
        oblq = 23.452294 - t * (0.0130125
                                  + t * (0.00000164 - t * 0.000000503))
        eccen = 0.01675104 - t * (0.00004180 + t * 0.000000126)

        # Convert to principle angles
        manom = (manom % 360.0)
	lperi = (lperi % 360.0)

        # Convert to radians
	r = (ra * 15.)/self.RADEG
	d = dec/self.RADEG
	manom = manom/self.RADEG
	lperi = lperi/self.RADEG
	oblq = oblq/self.RADEG

        # TANOM is the true anomaly (approximate formula) (radians)
        tanom = (manom + (2.0 * eccen - 0.25 * eccen**3.0) * math.sin(manom)
                 + 1.25 * eccen**2.0 * math.sin(2.0 * manom)
                 + 13./12. * eccen**3.0 * math.sin(3.0 * manom))

        # SLONG is the true longitude of the Sun seen from the Earth (radians)
	slong = lperi + tanom + math.pi

        # L and B are the longitude and latitude of the star in the orbital
	# plane of the Earth (radians)
        l, b = self. ast_coord(0., 0., -math.pi/2., math.pi/2. - oblq, r, d)

        # R is the distance to the Sun.
        rsun = (1.0 - eccen**2.0) / (1.0 + eccen * math.cos(tanom))

        # LTIM is the light travel difference to the Sun.
        # HJD is the heliocentric Julian Day
        ltim = -0.005770 * rsun * math.cos(b) * math.cos(l - slong)
	hjd = jd + ltim

        return ltim, hjd

    def ast_vr(self, ra1, dec1, v1, ra2, dec2):
        """Project a velocity vector in radial velocity along line of sight.

        :param ra1: Right ascension of velocity vector (hours)
        :param dec1: Declination of velocity vector (degrees)
        :param v1: Magnitude of velocity vector
        :param ra2: Right ascension of observation (hours)
        :param dec2: Declination of observation (degrees)
        """

        # Cartesian velocity components of the velocity vector.
        vx = v1 * math.cos((15. * ra1)/self.RADEG) * math.cos(dec1/self.RADEG)
        vy = v1 * math.sin((15. * ra1)/self.RADEG) * math.cos(dec1/self.RADEG)
        vz = v1 * math.sin(dec1/self.RADEG)

        # Direction cosines along the direction of observation.
        cc = math.cos(dec2/self.RADEG) * math.cos( (15. * ra2)/self.RADEG)
        cs = math.cos(dec2/self.RADEG) * math.sin( (15. * ra2)/self.RADEG)
        s  = math.sin(dec2/self.RADEG)

        # Project velocity vector along the direction of observation.
        v2 = (vx * cc + vy * cs + vz * s)
        return v2


    def ast_vorbit(self, ra, dec, epoch):
        """Radial velocity component of the Earth-Moon barycenter
        relative to the Sun.

        :param ra: RA of observation (hours)
        :param dec: DEC of observation (degrees)
        :param epoch: Julian epoch of observation
        """
        # T is the number of Julian centuries since J1900.
        t = (self.ast_julday(epoch) - 2415020.) / 36525.
        
        # MANOM is the mean anomaly of the Earth's orbit (degrees)
	# LPERI is the mean longitude of perihelion (degrees)
	# OBLQ is the mean obliquity of the ecliptic (degrees)
	# ECCEN is the eccentricity of the Earth's orbit (dimensionless)
        manom = 358.47583 + t * (35999.04975 - t * (0.000150 + t * 0.000003))
        lperi = 101.22083 + t * (1.7191733 + t * (0.000453 + t * 0.000003))
        oblq = (23.452294 - t * (0.0130125
                                + t * (0.00000164 - t * 0.000000503)))
        eccen = 0.01675104 - t * (0.00004180 + t * 0.000000126)

        # Convert to principle angles
        manom = (manom % 360.0)
	lperi = (lperi % 360.0)

        # Convert to radians
	r = (ra * 15.)/self.RADEG
	d = dec/self.RADEG
	manom = manom/self.RADEG
	lperi = lperi/self.RADEG
	oblq = oblq/self.RADEG

        # TANOM is the true anomaly (approximate formula) (radians)
        tanom = (manom + (2.0 * eccen - 0.25 * eccen**3.0) * math.sin(manom)
                 + 1.25 * eccen**2.0 * math.sin(2.0 * manom)
                 + 13./12. * eccen**3.0 * math.sin(3.0 * manom))

        # SLONG is the true longitude of the Sun seen from the Earth
	slong = lperi + tanom + math.pi

        # L and B are the longitude and latitude of the star in the
        # orbital plane of the Earth (radians)
        l, b = self.ast_coord(0., 0., -math.pi/2., math.pi/2 - oblq, r, d)

        # VORB is the component of the Earth's orbital velocity perpendicular
        # to the radius vector (km/s) where the Earth's semi-major axis is
        # 149598500 km and the year is 365.2564 days.
        vorb = ((2.0*math.pi / 365.2564) * 149598500.
                / math.sqrt(1.0 - eccen**2.)) / 86400.

        # V is the projection onto the line of sight to the observation of
        # the velocity of the Earth-Moon barycenter with respect to the
        # Sun (km/s).
        v = vorb * math.cos(b) * (math.sin(slong - l) - eccen * math.sin(lperi - l))
        return v


    def ast_vbary(self, ra, dec, epoch):
        """Radial velocity component of center of the Earth relative to
        to the barycenter of the Earth-Moon system.

        :param ra: RA of observation (hours)
        :param dec: DEC of observation (degrees)
        :param epoch: Julian epoch of observation
        """

        # T is the number of Julian centuries since J1900.
        t = (self.ast_julday(epoch) - 2415020) / 36525.

        # OBLQ is the mean obliquity of the ecliptic
        # OMEGA is the longitude of the mean ascending node
        # LLONG is the mean lunar longitude (should be 13.1763965268)
        # LPERI is the mean lunar longitude of perigee
        # INCLIN is the inclination of the lunar orbit to the ecliptic
        # EM is the eccentricity of the lunar orbit (dimensionless)
        # All quantities except the eccentricity are in degrees.

        oblq = (23.452294 - t * (0.0130125 + t * (0.00000164 - t
                                                  * 0.000000503)))
        
        omega = (259.183275 - t * (1934.142008 + t * (0.002078 +
                                                      t * 0.000002)))
        
        llong = (270.434164 + t * (481267.88315
                                   + t * (-0.001133
                                          + t * 0.0000019)) - omega)
        
        lperi = (334.329556 + t * (4069.034029
                                   - t * (0.010325
                                          + t * 0.000012)) - omega)
        em = 0.054900489
        inclin = 5.1453964

        # Determine true longitude.  Compute mean anomaly, convert to
        # true anomaly (approximate formula), and convert back to
        # longitude.  The mean anomaly is only approximate because
        # LPERI should be the true rather than the mean longitude of
        # lunar perigee.
        
        lperi = lperi/self.RADEG
        llong = llong/self.RADEG
        anom = llong - lperi
        anom = (anom + (2.0 * em - 0.25 * em**3.0) * math.sin(anom)
                + 1.25 * em**2.0 * math.sin(2.0 * anom)
                + 13./12. * em**3.0 * math.sin(3.0 * anom))
        llong = anom + lperi

        # L and B are the ecliptic longitude and latitude of the
        # observation.  LM and BM are the lunar longitude and latitude
        # of the observation in the lunar orbital plane relative to
        # the ascending node.

        r = (ra * 15)/self.RADEG
        d = dec/self.RADEG
        omega = omega/self.RADEG
        oblq = oblq/self.RADEG
        inclin = inclin/self.RADEG

        l, b = self.ast_coord(0.,0., -math.pi/2., math.pi/2. - oblq, r, d)
        lm, bm = self.ast_coord(omega, 0., omega - math.pi/2., math.pi/2. - inclin, l, b)

        # VMOON is the component of the lunar velocity perpendicular
        # to the radius vector.  V is the projection onto the line of
        # sight to the observation of the velocity of the Earth's
        # center with respect to the Earth-Moon barycenter.  The 81.53
        # is the ratio of the Earth's mass to the Moon's mass.

        vmoon = ((2.0*math.pi / 27.321661) * 384403.12040
                 / math.sqrt(1.0 - em**2.0) / 86400.)
        v = vmoon * math.cos(bm) * (math.sin(llong - lm) - em * math.sin(lperi - lm))
        v = v / 81.53

        return v

    def ast_vrotate(self, ra, dec, epoch, latitude, longitude, altitude):
        """ Radial velocity component of the observer relative to the
        center of the Earth due to the Earth's rotation.

        :param ra: Right Ascension of observation (hours)
        :param dec: Declination of observation (degrees)
        :param epoch: Epoch of observation (Julian epoch)
        :param latitude: Latitude (degrees)
        :param longitude: Latitude (degrees)
        :param altitude: Altitude (meters)
        """

        # LAT is the latitude in radians.
        lat = latitude/self.RADEG
        
        # Reduction of geodesic latitude to geocentric latitude
        # (radians).  Dlat is in arcseconds.
        dlat = (-(11.0 * 60.0 + 32.743000) * math.sin(2.0 * lat)
                + 1.163300 * math.sin(4.0 * lat) -0.002600 * math.sin(6.0 * lat))
        lat = lat + ((dlat / 3600.) / self.RADEG)

        # R is the radius vector from the Earth's center to the
        # observer (meters).  Vc is the corresponding circular
        # velocity (meters/sidereal day converted to km / sec).
        # (sidereal day = 23.934469591229 hours (1986))

        r = (6378160.0 * (0.998327073 + 0.00167643800 * math.cos(2.0 * lat)
                          - 0.00000351 * math.cos(4.0 * lat)
                          + 0.000000008 * math.cos(6.0 * lat))
             + altitude)
        
        vc = 2.0 * math.pi * (r / 1000.)  / (23.934469591229 * 3600.0)

        # Project the velocity onto the line of sight to the star.
        lmst = self.ast_mst(epoch, longitude)
        v = (vc * math.cos(lat) * math.cos(dec/self.RADEG)
             * math.sin(((ra - lmst) * 15.)/self.RADEG))
        
        return v

    def rvcorrect(self, vobs=0.):
        """ Compute the radial velocities.
        """

        # SOLAR INFORMATION
        vs = 20.0     # Sun velocity relative to the LSR
        ras = 18.0    #"18:00:00"
        decs = 30.0   #"30:00:00"
        eps = 1900.0

        # Determine epoch of observation and precess coordinates.
        epoch = self.ast_date_to_epoch(self.year, self.month, self.day, self.ut)
       
        ra_obs, dec_obs = self.ast_precess(self.ra, self.dec,
                                           self.ep, epoch)
        ra_vsun, dec_vsun = self.ast_precess(ras, decs, eps, epoch)
        t, self.hjd = self.ast_hjd(ra_obs, dec_obs, epoch)

        # Determine velocity components.
        self.vsol = self.ast_vr(ra_vsun, dec_vsun, vs, ra_obs, dec_obs)
        self.vorb = self.ast_vorbit(ra_obs, dec_obs, epoch)
        self.vbary = self.ast_vbary(ra_obs, dec_obs, epoch)
        self.vrot = self.ast_vrotate(ra_obs, dec_obs, epoch, self.latitude,
                                self.longitude, self.altitude)
        
        # Computing heliocentric velocity
        self.vhelio = self.vrot + self.vbary + self.vorb + vobs
        self.vlsr = self.vrot + self.vbary + self.vorb + self.vsol + vobs

        if not self.silent:
            print("VOBS: %f km.s-1"%vobs)
            print("HJD: %f"%self.hjd)
            print("VHELIO: %f km.s-1"%(self.vhelio))
            print("VLSR: %f km.s-1"%self.vlsr)
            print("VDIURNAL: %f km.s-1"%self.vrot)
            print("VLUNAR: %f km.s-1"%self.vbary)
            print("VANNUAL: %f km.s-1"%self.vorb)
            print("VSOLAR: %f km.s-1"%self.vsol)
            
        return self.vhelio, self.vlsr
