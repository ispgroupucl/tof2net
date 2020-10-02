""" .. currentmodule::tofnet.utils

    This module contains the utilities for different part of the code that didn't fit in another
    more clearly defined sub-module:


      * :mod:`config`: handles the config object based on the config file as well
        as the loading of functions from different modules
      * :mod:`counter`: counts the operations (GOPs) inside a neural network
      * :mod:`io`: reads and writes the data from and to disk
      * :mod:`metrics`: self-explanatory metrics and imports from ignite for common ones
      * :mod:`notebook_utils`: various utilities for ipywidgets to have interactive notebooks
      * :mod:`outputs`: contains the Output objects
      * :mod:`pointcloud`: contains various functions that are more high-level than :mod:`tofnet.pointcloud.utils`
        both files are complementary.
      * :mod:`print_utils`: helper functions to help printing with tqdm on a tmux terminal
      * :mod:`torch_utils`: helper function for pytorch batch aggregation
"""