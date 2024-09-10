"""Neural network model-based treatment plan."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.base import TreatmentPlan
from pyanno4rt.gui import GraphicalUserInterface
from pyanno4rt.tools import snapshot

# %% Initialization

tp_nn = TreatmentPlan(

    configuration={
        "label": "Head-and-neck-neural-network",
        "modality": "photon",
        "imaging_path": "../external/HNC_patientData.mat",
        "dose_matrix_path": "../external/HNC_photonDij.mat",
        "dose_resolution": [5, 5, 5]
    },

    optimization={
        "components": {
            "PAROTID_LT": [
                {
                    "type": "objective",
                    "instance": {
                        "class": "Squared Overdosing",
                        "parameters": {
                            "maximum_dose": 25.0,
                            "embedding": "active",
                            "weight": 100.0
                        }
                    }
                },
                {
                    "type": "objective",
                    "instance": {
                        "class": "Neural Network NTCP",
                        "parameters": {
                            "model_parameters": {
                                "model_label": "neuralNetworkNTCP",
                                "model_folder_path": "/home/tim/Schreibtisch/Promotion/Doktorarbeit/pyanno4rt/snaps//Head-and-neck-neural-network/neuralNetworkNTCP"
                            },
                            "embedding": "active",
                            "weight": 10.0,
                            "link": ["PAROTID_RT"],
                        }
                    }
                }
            ],
            "PAROTID_RT": {
                "type": "objective",
                "instance": {
                    "class": "Squared Overdosing",
                    "parameters": {
                        "maximum_dose": 25.0,
                        "embedding": "active",
                        "weight": 100.0,
                    }
                }
            },
            "PTV70": {
                "type": "objective",
                "instance": {
                    "class": "Squared Deviation",
                    "parameters": {
                        "target_dose": 70.0,
                        "embedding": "active",
                        "weight": 1000.0,
                    }
                }
            },
            "SKIN": {
                "type": "objective",
                "instance": {
                    "class": "Squared Overdosing",
                    "parameters": {
                        "maximum_dose": 30.0,
                        "embedding": "active",
                        "weight": 800.0,
                    }
                }
            },
            "PTV63": [
                {"type": "objective",
                 "instance": {
                     "class": "Squared Deviation",
                     "parameters": {
                         "target_dose": 63.0,
                         "embedding": "active",
                         "weight": 1000.0}
                     }
                 },
                {"type": "objective",
                 "instance": {
                     "class": "Neural Network TCP",
                     "parameters": {
                         "model_parameters": {
                             "model_label": "neuralNetworkTCP",
                             "model_folder_path": "/home/tim/Schreibtisch/Promotion/Doktorarbeit/pyanno4rt/snaps//Head-and-neck-neural-network/neuralNetworkTCP"
                             },
                         "embedding": "active",
                         "weight": 10.0}
                     }
                 }],
            },
        "method": "weighted-sum",
        "solver": "scipy",
        "algorithm": "L-BFGS-B",
        "max_iter": 500,
        "tolerance": 1e-3
    },

    evaluation={
        "display_segments": [],
        "display_metrics": []
        }

    )

# %% Compose

tp_nn.compose()

# %% Snapshot

# snapshot(tp_nn, './snaps/')

# %% GUI

gui = GraphicalUserInterface()
gui.launch(tp_nn)
