# Experimenting Framework for Training and Evaluating SAC-based Multi-Agent RL Algorithms for Vehilce Platooning in SUMO

The SAC implementation is due to Yarats, Denis and Kostrikov, Ilya (https://github.com/denisyarats/pytorch_sac).

## Requirements
If you want your environment to run in SUMO you first need to install Flow (https://github.com/flow-project/flow) which lets you create RL-compatible SUMO environments and therefore acts as an interface between SUMO and the RL algorithms. Detailed instructions on the installation of Flow and SUMO can be found here: https://flow.readthedocs.io/en/latest/flow_setup.html . A predefined platooning environment for Flow can be taken from this repository: https://github.com/RGR-repo-cloud/Flow-based_RL-framework_for_Platooning_in_SUMO , which can also be cloned instead of the original Flow repository but must be renamed to "flow" addtitionally. This framework is of course also compatible with other gym (https://github.com/openai/gym) environments. For this to work you just need to adapt the instantiation of the environment which is located in the constructor of the Workspace class.

## Instructions
To train or evaluate an algorithm you can specify the parameters of the SAC algorithm in the "sac.yaml" file and other general parameters for running an algorithm in the "run.yaml" file. Then just execute the "run.py" file.

## Results and Checkpoints
Data logged during a training or evaluation run as well as checkpoints if specified are stored in the "exp" directory. 

