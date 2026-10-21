## Multi-Modal Transfer for Cold hardiness
Developed by Will Solow and Sandhya Saisubramanian
Oregon State University, Corvallis, OR

Correspondance to Will Solow, soloww@oregonstate.edu 

### Installation

1. Create a virtual environment 
* `conda create -n mmtransfer python=3.12`
* `conda activate mmtransfer`

2. Clone this repository
* `git clone https://github.com/Intelligent-Reliable-Autonomous-Systems/Multi-Modal-Transfer.git`

3. Install dependencies with pip
* `pip install -e model_engine`
* `pip install -e mm_transfer` 

### Running experiments
1. See available data in _data/
2. Configure train.yaml file in _train_configs/
3. Run experiment with `python3 -m trainers.train_model --config train.yaml`