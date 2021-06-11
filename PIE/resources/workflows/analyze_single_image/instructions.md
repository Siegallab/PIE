This page runs PIE on a single colony image. The results are:

* A jpeg of the original image, overlaid with the contours of the colony mask, and
* a table of detected cells and their areas and positions.

The inputs are:

* `Input Image File`: The image to be analyzed

* `Image Type`: The type of micrscopy used to generate images.

* `Hole Fill Area`: The area (in pixels) of the largest size hole to fill in colony masks after image analysis. For low-res yeast imaging, we recommend leaving this field empty (i.e. all the holes in the colonies get filled).

* `Cleanup`: Whether or not to perform 'cleanup' of spurious pieces of background attached to colonies. We recommend trying PIE both with and without cleanup on a set of sample images. You can see the Li, Plavskin *et al.* paper for details.

* `Max. Proportion`: Maximum proportion (0-1) of the perimeter of a PIE-detected gradient object ('PIE piece') that may be non-adjacent to another PIE piece to avoid being removed during 'cleanup' steps (only used if `cleanup` is *True*). The default value of 0.75 works well for 10x yeast images.
