# Experimenting Framework for Training and Evaluating SAC-based Multi-Agent RL Algorithms for Vehilce Platooning in SUMO

The SAC implementation is due to Denis Yarats and Ilya Kostrikov (https://github.com/denisyarats/pytorch_sac).

## Requirements
First, you need to install SUMO (https://github.com/eclipse-sumo/sumo) and the Flow framework (https://github.com/flow-project/flow). Detailed instructions on the installation of Flow and SUMO can be found here: https://flow.readthedocs.io/en/latest/flow_setup.html . Instead of the original Flow framework, a version modified specifically for vehicle platooning must be installed, though, which can be found here: https://github.com/RGR-repo-cloud/Flow-based_RL-framework_for_Platooning_in_SUMO . Note, that it must be renamed to "flow", additionally.

## Instructions
To train or evaluate an algorithm you can specify the parameters of the SAC algorithm in the "sac.yaml" file and other general parameters in the "run.yaml" file. Then just execute the "run.py" file.
If you want to evaluate a classic control algorithm specifications can be made in the "external_control_eval.yaml' file and the run file is "external_control_eval.py".

Functionality for calculating training and evaluation statistics as well as plotting is provided via Jupyter Notebooks.

## Results and Checkpoints
Data logged during a training or evaluation run as well as checkpoints if specified are stored in the "exp" directory.
Control data obtained during an evaluation of a classic controller is saved in the "ext_eval" directory.

