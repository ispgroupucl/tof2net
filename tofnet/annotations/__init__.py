""" This module contains all files necessary for the creation of new data.
    The mandatory data consists of the pcd and conf folders. Based on those,
    the different functions can create different input-types, segmentation masks
    or other configs.

    The files posses:

        * a :func:`generate_sample` function that, based on a given style-string,
          creates the desired transformation of the given sample. (optional)
        * a :func:`generate` function that manages the read and write respectively
          before and after the transformation done by generate_sample (mandatory)

"""