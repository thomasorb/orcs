.. _known_issues:

Known Issues
============

Filter fit
----------

SITELLE filters show slight changes in their bandpass with the (x,y)
position. This changes is generally a shift of the filter curve but,
especially in the corners, the filter curve cannot be modeled any more
with only a wavelength shift. The best way to model the filter would
be to measure the filter curve everywhere in the field of view and get
a 3d filter curve. Up to now we only have a few measures in the center
and the corners and the filter model is thus not behaving very well in
the corners of the field. If you experience such a problem, a good
workaround is to set the option 'nofilter=True' in the fitting
function. The drawback is that you won't be able to fit emission lines
on the border of the filter.