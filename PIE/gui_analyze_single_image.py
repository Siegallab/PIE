"""
Run single image analysis from a GUI that is rendered by flowServ using
streamlit.

Streamlit requires a main file to run the app. This file will be 'executed'
from the command line interface to start the GUI. Base on:
https://discuss.streamlit.io/t/running-streamlit-inside-my-own-executable-with-the-click-module/1198/4
"""

import os

from flowserv.client.gui.app import main


if __name__ == '__main__':
    # Get the package path for the workflow template.
    dirname = os.path.dirname(__file__)
    source = os.path.join(dirname, 'resources', 'workflows', 'analyze_single_image')
    # Run the streamlit GUI for the workflow template.
    main(source=source, specfile=None, manifestfile=None, name='analyze_single_image')
