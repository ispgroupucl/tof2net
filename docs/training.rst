================
Training a Model
================

In order to train a model, you need a configuration file. This configuration
file should be written in the `TOML language <https://github.com/toml-lang/toml>`_.

A typical example of a config file can be found in the `Config Example`_ section.

You should always start from an existing config file, because a lot of information
is always the same (such files are available in ``configs/``).

Once you have a valid configuration, training is very easy. Simply run::
   
   ./tof train configs/config.toml

This will create a new directory in the :file:`logs/` directory (by default). This
directory will contain either directly the model or will contain directories containing
the models.

Configuration file
==================

The config file is divided into sections, denoted by ``[section]``. Some sections
accept more than one instance, and will create a list of objects. Such sections
are denoted by ``[[section]]``.

The important parts of the config will be detailed in the next subsections.

Network
-------

This section relates to the definition of the model/network. The recommended
type is ``conf_unet`` which will allow the precise definition of the network
(using ``network.architecture``).

The value that will probably have to be changed is the ``input_format``. The input
format contains the types (and not the exact names) of the inputs that the network
expects. For example, if you need intensity, height and normal information, the 
config will look like::

   input_format = ['image','depth','depth']

This is because both the normal information and height information have the depth
type (despite the big difference between the two, they are generated in the same
way).

If you have a higher resolution image, you might want to add another layer to 
the architecture. To do this, you should change the definition of 
``[network.architecture]``:

.. code-block:: toml

   [network.architecture]
   first = 32
   enc.width = [64, 72, 96, 128]
   enc.repeat = [2, 2, 3, 4]
   dec.width = [72, 64, 64]
   dec.repeat = [2, 2, 1]

might become 

.. code-block:: toml

   [network.architecture]
   first = 32
   enc.width = [64, 64, 72, 96, 128]
   enc.repeat = [1, 2, 2, 3, 4]
   dec.width = [72, 64, 64, 64]
   dec.repeat = [2, 2, 2, 1]



Most other parameters won't need to be changed, but here are their explanations:

:big_drop and small_drop:
   Control the dropout in different parts of the network

:bn:
   Controls the batch normalization

:block:
   Changes the type of the building block that is used for the network. possible
   value are (todo)

:last_act:
   Changes the activation function of the last layer. Do not change !

:se_ratio:
   A squeeze-and-excite parameter controlling the SE ratio.

:multi_input:
   Should be true


Training
--------

The default (and only) type available to training is ``default``. The training
section controls how the training happens, and is useful if you need to change
batch size the number of epochs, the learning rate, etc.

:batch_size:
   the batch size. Is especially important on GPU training. When changing the batch
   size, you should always change the learning rate to balance out the rate of 
   learning.

:accumulation:
   Accumulation is related to the batch size. It is useful in case there is not 
   a lot of memory on GPU and the batch size is not high enough. For example a
   batch size of 16 and accumulation at 2 will have an effective batch size of 32.

:loss:
   Type of loss function to use. Loss definitions are available in :mod:`tofnet.utils.losses`.
   Most are automatically `imported from pytorch <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.

:n_epochs:
   Number of epochs. One epoch is a pass through all the training data. This value
   should be modified if you change the number of images in the training set.
   For example if you have 1000 images or 10 images you should probably have a different
   number of epochs. Because we always take the best model (on the validation set),
   this value can be larger than needed.

Training optimizer
::::::::::::::::::

This section inside the training section controls the optimizer, and therefore
the learning rate. Some parameters depend on the type, and for this it is always
best to look at the different optimizers available in :mod:`tofnet.utils.optimizers`
and in the `pytorch documentation on optimizers <https://pytorch.org/docs/stable/optim.html>`_.

:type:
   The optimizer type. In most cases, we use AdamW, a modified version of the Adam
   optimizer with correct weight-decay.

:lr:
   The learning rate. **This value should be searched when training on a new set of
   data**.

Training Scheduler
::::::::::::::::::

The scheduler can be chosen with the type of this section. More information for 
the available schedulers in the `pytorch documentation  on schedulers <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.


Dataset
-------
.. currentmodule:: tofnet.data

The dataset consists of three sections, ``[dataset]``, ``[val_dataset]``, and
``[test_dataset]``, each controlling the configuration of the training, validation
and testing sets respectively. Because those contain redundant information, the
val_dataset copies missing information from dataset and test_dataset copies 
information from val_dataset and dataset.

:type:
   a class from :mod:`tofnet.data.datasets`. Probably :class:`datasets.Default`. Additional
   parameters are defined in the api documentation.

:name:
   The name of the directory containing the dataset.

:dtypes:
   the types that should be imported and are directly available from the file system.
   Other dtypes will be automatically added with the DatasetMaker.


:classes:
   For segmentation, the labels of the classes, without the background, in order.
   Empty strings will be removed from the annotation and the resulting annotation 
   will be relabeled. for example, suppose you have a dataset with floor, bed
   and person, but don't want to segment the floor, you can have the following config::

      classes = ['', 'bed', 'person']

   it will change the numbers of the classes to 1 for bed and 2 for person 
   (and always 0 for background).

:dimensions:
   The dimensions to resize, in order to input the image at the correct dimension
   in the model. It should be divisible by :math:`2^S` with S the number of stages in
   the model.

:mask.weights:
   The weights of the different classes. This list should be one element larger
   than the :data:`dataset.classes` list, and contain a weight multiplier for every label
   including background.

Strategy
--------
.. currentmodule:: tofnet.training.strategies

The strategy section is a bit peculiar in its usage. It contains all classes that
will change the configuration itself and have side-effects. The most useful strategy
is the :class:`tofnet.training.strategies.DatasetMaker`. From a dataset containing
only pointclouds and :file:`kaspard.conf`'s. Because most of this information takes
a long time to build, the directories made by :class:`DatasetMaker` are cached.

Another useful strategy is :class:`LeaveOneOut` which uses common file name prefixes 
in order to create different models with a K-Fold. For every iteration of the 
K-Fold, the i-th prefix is used for testing and the i+1-th prefix for validation,
all the other files are used for training. The prefixes can for example be the names
of the institutions or arbitrary prefixes. (In our case , we used :file:`Fold{X}`).

To be able to search for some hyper-parameters, you can use :class:`GridSearch`.
This will add a magic keyword :data:`hyper` that you can add to any field, separated
by a dot. You can then change the value into a list of values. For example:

.. code-block:: toml

   [[strategy]]
   type = 'GridSearch'
   best_select = 'max'

   [training.optimizer]
   lr.hyper = [0.01, 0.001, 0.0001]

   [dataset]
   mask.weights.hyper = [[0.5, 1.0, 1.5], [0.1, 100, 2]]

In this case, :math:`3*2` models will be created, and the one with the best value
on the validation set will be selected. If you combine this with :class:`LeaveOneOut`
you will try a lot more combinations.

Saves
-----

Decides where to save the models, and what to use to decide the best model.

:save_best_only:
   Boolean if you want to keep the last model or only the best one.

:path:
   path where you should save the model, by default, :file:`logs/` is used.

:monitor:
   what value to monitor in order to choose the best model. This variable is also
   used for the :class:`GridSearch`

Config Example
==============

.. code-block:: toml

   version = 1
   type = 'v4_H_hN_mIn5'
   augmentation_function = 'default' # For backward compatibility

   [network]
   type = 'conf_unet'
   bn = true
   block = 'residual'
   conv_transpose = false
   last_act = 'linear'
   big_drop = 0.4
   small_drop = 0.2
   se_ratio = 16
   input_format = ['image', 'depth', 'depth']
   output_format = ['mask']
   multi_input = true

   [network.architecture]
   first = 32
   enc.width = [64, 72, 96, 128]
   enc.repeat = [2, 2, 3, 4]
   dec.width = [72, 64, 64]
   dec.repeat = [2, 2, 1]

   [training]
   type = 'default'
   batch_size = 16
   accumulation = 2
   loss = ['CrossEntropyLoss']
   n_epochs = 80

   [training.optimizer]
   lr = 0.001
   type = 'AdamW'
   weight_decay = 1e-5

   [training.weight_init]
   sampling = 'kaiming'
   distribution = 'normal'
   fan_mode = 'fan_in'

   [training.scheduler]
   type = 'MultiStepLR'
   milestones = [23, 40]
   gamma = 0.1

   [val_dataset]
   type = 'Default'

   [test_dataset]
   type = 'Default'
   resize = 'later'
   dimensions = [1, 128, 192]

   [dataset]
   type = 'Default'
   dtypes = ['image']
   resize = 'crop3'
   name = 'kaspard_pcdv4'
   classes = ['floor', 'bed', '', '']
   channels_first = true
   dimensions = [1, 128, 128]
   mask.weights = [0.5, 1.0, 1.5]

   [augmentation]
   horizontal_flip = 0.5
   zoom_range = 0.2

   [saves]
   save_best_only = true
   path = 'logs/'
   monitor = 'val_mask_miou'

   [[strategy]]
   type = 'DatasetMaker'
   dtype = 'depth'
   style = 'height'
   format_type = 'input'

   [[strategy]]
   type = 'DatasetMaker'
   dtype = 'depth'
   style = 'normalsxyzheight'
   format_type = 'input'

   [[strategy]]
   type = 'DatasetMaker'
   dtype = 'mask'
   style = 'normal'
   format_type = 'output'

   [[strategy]]
   type = 'LeaveOneOut'
   prefixes = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6', 'Fold7']
   infix = '--'





API documentation
=================

Models
------
.. automodule:: tofnet.models
   :members:

Losses
------
.. automodule:: tofnet.training.losses
   :members:

Strategies
----------
.. automodule:: tofnet.training.strategies
   :members: