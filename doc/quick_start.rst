Quick start Guide
#################

.. contents::

Step 1: Create your option file
-------------------------------

Like ORBS, ORCS uses option files to load all the parameters. Here's
en example.

.. literalinclude:: option_file.orc
   :language: python

**REQUIRED** parameters
~~~~~~~~~~~~~~~~~~~~~~~
:option:`INCLUDE` Include ORBS option file.

:option:`DIRSPEC` Path to the spectrum directory. This directory can
        be created from a full 3d cube by using the 'orb-unstack'
        function of ORB.

:option:`WAVENUMBER` Must be 0 if the cube axis is in wavelength or 1
        if the cube axis is in wavenumber.

:option:`WAVE_CALIB` Must be 0 if the cube is not wavelength
        calibrated and 1 if the cube is wavelength calibrated.

:option:`APOD` Give the apodization (1.0 for unapodized cube)

:option:`LINES` Emission Lines names, enter an integer or a float to
        use a non_recorded emission line. See
        :py:class:`orb.core.Lines` to get a list of the available
        lines.

:option:`COV_LINES` Covarying lines. Give the same number to the lines
        from the same atom, or more generally for the lines of the
        same velocity.

:option:`OBJECT_VELOCITY`  Object velocity in km.s-1

:option:`POLY_ORDER` Order of the polynomial used to fit continuum

:option:`ROI` Region of interest : Xmin,Xmax,Ymin,Ymax. Must be integers.

:option:`SKY_REG` Sky regions ds9 file path (Regions border must be in
        pixel coordinates, region shape can be a circle, a box or a
        polygon)

:option:`CALIBMAP` Calibration map path

:option:`DATE` Observation date (UT)

:option:`HOUR_UT` Observation hour (UT)

Step 2: Run it
--------------

.. code-block:: console

  orcs option_file.orc start
