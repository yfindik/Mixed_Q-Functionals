# Mixed Q-Functionals: Advancing Value-Based Methods in Cooperative MARL with Continuous Action Domains

This repository is the official implementation of Mixed Q-Functionals: Advancing Value-Based Methods in Cooperative MARL with Continuous Action Domains. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Note

While first training for Mixed Q-functionals, 
you may encounter an error related to the 'cpprb' library.
To resolve this issue, please replace its 'util.py' file 
with the one provided in the 'modification_cpprb' folder.




## Training

An example command to train the models in the paper is as follows:
```setup
python experiments/[environment_name]/training_[algo_core_name].py 
--experiment_name results/final/2-walker-sample10k 
--env_name walker 
--is_sisl_env 
--n_walkers 2 
--max_episode_len 500  
--nb_runs 10 
--max_episode 30000 
--max_step 1500000 
--update_step 50 
--saving_frequency 10000 
--policy_type gaussian 
--ma_type mix_sum 
--num_layers 2 
--gamma 0.99
```
Here, **environment_name** can be either 'mpe' or 'walker', and **algo_core_name** is either 'mqf' or 'ddpg'.

For a specific scenario under the mpe environment, the hyperparameter **--env_name** should be the name of the mpe scenario with the **--is_not_gym** flag.

For the walker environment, the hyperparameter **--n_walkers** should be 1 or 2 with the **--is_sisl_env** flag.

We have created service scripts for training in the background. 
You can follow the guideline given under the services directory named 'useful_commands.txt'. 
The hyperparameters used to run the experiments can be found in the respective service scripts. 
The 'ExecStart' command within the service scripts can be directly used to train the models.


## Animation

An example command to animate the agents behavior after training, as follows:
```setup
python experiments/[environment_name]/animation_[algo_core_name].py 
--ma_type mqf 
--scenario 3predator_1prey_ICPP 
--seed 0 
--nb_test 10 
--step 1 
--render
--path_to_model [experiment_name]/[model_dir_name]
```
Here, **environment_name** can be, again, either 'mpe' or 'walker', and **algo_core_name** is either 'mqf' or 'ddpg'.



## Analysis

For the analysis of the models in terms of rewards, success rate, and other metrics, 
please refer to the other [repository](https://drive.google.com/drive/folders/1c3f0vD0tV7AGRSN56kS6iSxGk9t98NSl). 
You will find the scripts for the figures in the main paper and appendix, 
as well as the logs in each experiment's folder.


## ToDos

- Share the latest trained models for the algorithms.




