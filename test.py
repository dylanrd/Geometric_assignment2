# This should be invoked with the following command line (or equivalent)
# blender --background --python /home/jcampolattaro/Documents/geometric-data-proc/test.py -- --verbose
import os
import sys

# Blender will actually run this in another directory, so we need to make sure everything is available to import
sys.path.append(os.path.dirname(__file__))

# Make sure we have the packages we need
import pip
pip.main(['install', '-r', f'{os.path.dirname(__file__)}/requirements.txt'])

# Dealing with contested command line parameters
# see: https://blender.stackexchange.com/questions/267812/blender-doesnt-recognize-python-as-a-command-line-argument
argv = [__file__]
if "--" in sys.argv:
    argv += sys.argv[sys.argv.index("--") + 1:]

# Import your package & run its unit tests
from assignment3 import *
unittest.main(argv=argv)
