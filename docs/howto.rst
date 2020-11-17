.. tofnet documentation master file, created by
   sphinx-quickstart on Wed Sep 23 13:29:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================
Various How To's
================

How to change image resolution
==============================

In the configuration file, you have two or three sections you should change:

In dataset, you have to change the resolution (for the training set and for 
the test and validation set) :

.. code-block:: toml

   [dataset]
   # (...)
   dimensions = [1, 256, 256]

   [val_dataset]
   dimensions = [1, 256, 320]

In the network you might want to change the number of levels of the U-Net
structure:

.. code-block:: toml

   [network.architecture]
   first = 32
   enc.width = [64, 64, 72, 96, 128]
   enc.repeat = [1, 2, 2, 3, 4]
   dec.width = [72, 64, 64, 64]
   dec.repeat = [2, 2, 2, 1]



How to add a class
==================
.. currentmodule:: tofnet.annotations.segmentation

The first step for adding a class is annotation. Two different ways of annotation are possible,
either you add an object with the ``kaspard.conf`` annotator or you create a segmentation mask
directly.

If you choose to use the ``kaspard.conf`` method, you will have to modify the 
:mod:`tofnet.annotations.segmentation` file. The :func:`segment` method will have to become
a loop which goes over the different objects in the room. :func:`find_segmentation`
will need to take a new parameter in order to have different class numbers 
(floor=1, bed=2, chair=3, ...).

Once the inputs are handled, you can change the configuration to add the new classes.

Let's say that you added the chair class, your configuration changes would look like:

.. code-block:: toml

   [dataset]
   classes = ['floor', 'bed', 'chair']

Non-exclusive classes
---------------------

If you want to have some classes that can be activated at the same time as other
classes (for example person and bed), you will need to change the model to have
different outputs:

.. code-block:: toml

   [network]
   output_format = ["mask", "mask"]

   [[strategy]]
   type = "DatasetMaker"
   dtype = "mask"
   style = "normal"
   format_type = "output"
   
   [[strategy]]
   type = "DatasetMaker"
   dtype = "mask"
   style = "person"
   format_type = "output"

.. note::

   Notice the style of the second :class:`DatasetMaker`, it is ``person`` instead of
   ``normal``, and so you should change the :func:`generate_sample` function to
   add this new style.

How to specialize to a room
===========================
.. currentmodule:: tofnet.training.strategies


The simplest way to specialize to a room is to use the :class:`Specialize` 
strategy. This strategy should be used in place of the :class:`RandomSplit`
strategy.

.. code-block:: diff

     [[strategy]]
   - type = "RandomSplit"
   + type = "Specialize"
   + prefixes = ["newroom1", "newroom2"]
   + infix = "_"
   + mult = 16

The files of the new rooms should have a prefix different from all other files in
the training set : (with infix from config)

.. code-block:: console
   
   name@home$ ls
   room1_0  room2_0  newroom1_0  newroom2_0
   room1_1  room2_1  newroom1_1  newroom2_1


How to re-use weights for training
==================================

If you want to re-use the weights from a model for training, the simplest way is
to add it in the network :

.. code-block:: toml
   
   [network]
   pretrained_weights = "path/to/model"

How to use Grid Search
======================

To search for different values of parameters (for example learning rate), the easiest
way is to use the grid-search strategy :

.. code-block:: toml

   [[strategy]]
   type = "GridSearch"
   best_select = "max"

   [training]
   optimizer.lr.hyper = [1e-2, 1e-3, 1e-4]

   [[network.architecture.hyper]]
   first = 32
   enc.width = [64, 64, 72, 96, 128]
   enc.repeat = [1, 2, 2, 3, 4]
   dec.width = [72, 64, 64, 64]
   dec.repeat = [2, 2, 1, 1]

   [[network.architecture.hyper]]
   first = 32
   enc.width = [64, 72, 96, 128]
   enc.repeat = [2, 2, 3, 4]
   dec.width = [72, 64, 64]
   dec.repeat = [2, 2, 1]


This grid-search will try 3 learning rates and 2 network architectures. This means
that 6 models will be created, and the best of the 6 will be kept, while the others
will be removed automatically.

How to use Leave-One-Out (Cross-validation)
===========================================

If you ever need cross-validation, you should replace the default :class:`RandomSplit`
with :class:`LeaveOneOut`:

.. code-block:: toml

   [[strategies]]
   type = "LeaveOneOut"
   prefixes = ["room1", "room2", "room3"]
   infix = "_"

You can do this splitting at any level you like : it's recommended to not have too
many splits, but for example to split up multiple homes together (and add a common
prefix to the file names).

How to add an input (greyscale)
===============================

If you want to add a new input, you can choose to add a new directory manually
(easy) or add a :class:`DatasetMaker` that reads the pcd files automatically (convenient).

For both solutions, you will need to add a line to the input format:

.. code-block:: toml

   [network]
   input_format = ["image", "image", "depth", "depth"] # with DatasetMaker
   input_format = ["image", "greyscale", "depth", "depth"] # with manual directory


If you create the directory manually, you have to instruct the dataset to read the
directory: 

.. code-block:: toml

   [dataset]
   dtypes = ["image", "greyscale"]

If you use the :class:`DatasetMaker` method, you have to add its strategy: 

.. code-block:: toml

   [[strategy]]
   type = "DatasetMaker"
   dtype = "image"
   style = "greyscale"
   format_type = "input"

And of course you will need to add a function reading the greyscale information
inside :mod:`tofnet.annotations.image`.


How to use/create semantic annotation tool
==========================================

The experimental semantic annotation tool is available in :file:`annotator/`.

It's direct use is not recommended, but reading the source code might be beneficial.

It basically uses the :func:`cv2.grabCut` function, passing it both intensity, depth
and height in the three color channels of an image.

How to add people
=================

See `How to add a class`_ and particularily `Non-exclusive classes`_.