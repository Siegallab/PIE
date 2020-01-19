#!/usr/bin/python
import argparse
import sys

PIE_option_parser = argparse.ArgumentParser(description="Run PIE code")
PIE_option_parser.add_argument('--input_dir', type=str,
	help='Directory containing image files to process')
PIE_option_parser.add_argument('--output_dir', type=str,
	help="""
	Directory in which output folders containing output results will be
	created
	""")
PIE_option_parser.add_argument('--im_suffix', type=str,
	help='Image file names should be formatted as t[#][im_suffix].tif')
PIE_option_parser.add_argument('--inter_timepoint_time', type=str,
	help='How many seconds pass between each timepoint')


PIE_option_parser.add_argument('--bitdepth', type=int, const=11, default=11,
	help='bitdepth of images')
PIE_option_parser.add_argument('--minimum_growth_time', type=int, default=4,
	help="""
	smallest number of timepoints in which an object needs to increase in
	size in order to be considered a colony; important for removing objects
	like debris from analysis
	""")
PIE_option_parser.add_argument('--settle_frames', type=int, default=1,
	help="""
	the latest timepoint a colony can be missing and still be counted in the
	analysis this is important if there is some initial shifting of colony
	positions (e.g. due to microscope plate heat-induced expansion) in early
	timepoints; settle_frames should be ~ the number of images it takes for
	the position of each colony to 'settle'
	""")
PIE_option_parser.add_argument('--satellite_distance', type=int, default=60,
	help="""
	(distance in pixels)
	if a colony looks like it split in two (e.g. cells moving or image
	analysis accidentally missing a piece of colony), how close does the
	'satellite' colony have to be to the original in order to be counted
	as part of the original?
	""")



#   timepoint_list: list of timepoints that were imaged in input_dir

%   cleanup: whether or not to do 'cleanup' of spurious pieces of
%       background attached to colonies (we recommend trying both with
%       and without cleanup; you can see the Li, Plavskin et al. paper
%       for details)
%   min_unexposed_edge:  parameter related to proportion of its
%       perimeter of a segment that needs to be touching a colony to
%       avoid being 'cleaned up'; 0.25 seems to work for 10x yeast imgs
%       only used if cleanup is true
%   hole_fill_area: what is the area (in pixels) of the largest size
%       hole to fill in colony overlays after image analysis?
%       For low-res yeast imaging, we recommend setting this value to
%       inf, i.e. all the holes in the colonies get filled



if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print args.message


import subprocess

# check matlab path

subprocess.check_output('getconf _NPROCESSORS_ONLN',shell=True)
except subprocess.CalledProcessError:

if __name__ == '__main__':
	# check matlab path and add it if necessary