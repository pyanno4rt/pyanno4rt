"""
Python-based advanced numerical nonlinear optimization for radiotherapy \
(pyanno4rt) module.

==================================================================

pyanno4rt is a Python package for conventional and outcome prediction \
model-based inverse photon and proton treatment plan optimization, including \
radiobiological and machine learning (ML) models for tumor control \
probability (TCP) and normal tissue complication probability (NTCP).

This module aims to provide methods and classes for the import of patient \
data from different sources, the individual configuration and management of \
treatment plan instances, multi-objective treatment plan optimization, \
data-driven outcome prediction model handling, evaluation, and visualization.

It also features an easy-to-use and clear graphical user interface.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from pyanno4rt import (
    base, datahub, dose_info, evaluation, gui, input_check, learning_model,
    logging, optimization, patient, plan, tools, visualization)

__all__ = ['base',
           'datahub',
           'dose_info',
           'evaluation',
           'gui',
           'input_check',
           'learning_model',
           'logging',
           'optimization',
           'patient',
           'plan',
           'tools',
           'visualization']
