==========
Prediction
==========

How to predict
==============
.. currentmodule:: tofnet.predict

Prediction happens in three steps:

1) Initialization (:meth:`tofnet.predict.init`)
2) Floor segmentation and calibration (:meth:`tofnet.predict.predict_floor`)
3) Bed segmentation and localization (:meth:`tofnet.predict.predict_object`)

In order to predict you need :

1) A model, which is a directory containing the model parameters and the model
   configuration
2) Pointcloud(s)

Once you have this information, you only need to follow the three steps of 
prediction.

1) :func:`init` with the name of the directory containing the model;
2) :func:`predict_floor` with the result of the :func:`init` function, an empty dictionary
   and the pointcloud;
3) :func:`predict_object` with the result of :func:`init`, the result of 
   :func:`predict_floor` and the pointcloud.

API documentation
=================

.. automodule:: tofnet.predict
   :members:
