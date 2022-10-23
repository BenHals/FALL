.. FALL documentation master file, created by
   sphinx-quickstart on Mon Oct 10 19:01:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*************************************************
FALL - Framework for Adaptive Life-Long Learning
*************************************************

FALL is designed to enable both practitioners and researchers to develop and implement machine learning systems for streaming data, in the presence of changing conditions.
FALL is a framework for building adaptive learning systems capable of incrementally learning from streaming data, and adapting to changing conditions.
FALL is modular, based around a standard architecture for adaptive learning. This enables existing methods to be easily recreated, as well as novel methods to be easily implemented without re-implementing standard components.

.. image:: https://github.com/BenHals/FALL/blob/main/adaptation.gif
  :width: 800
  :alt: Alternative text

Installation
============

``pip install fall-ml==0.0.1``

Documentation
=============

Please refer to the package `homepage <https://benhalstead.dev/FALL/>`_ for more information and documentation.

Examples
========

Please refer to the binder `link <https://mybinder.org/v2/gh/BenHals/FALL/HEAD?labpath=examples%2F>`_ for interactive examples. 

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   fall.adaptive_learning.rst
   fall.classifiers.rst
   fall.concept_representations.rst
   fall.data.rst
   fall.repository.rst
   fall.states.rst
   fall.rst
   modules.rst


