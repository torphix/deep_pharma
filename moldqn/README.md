# Deep Pharma A molecular generation and optmisation toolkit


# Data



# DQN For molecular optimisation:
    - If using pharmacophore add an enforce substrcuture rule preventing pharmacophore modification

# Protein to pocket prediction 

# Toxcitiy prediction

# Streamlit gui for interacting with the various models


# Citation:

```
This repo implements and modifies on several papers given below:
    - Dataset:   Brylinski, M., & Shi, W. (2022, May 13). Pocket2Drug. https://doi.org/10.17605/OSF.IO/QACWJ

    - Graphsite: 

    - Pocket2Drug: An encoder-decoder deep neural network for the target-based drug design
        Wentao Shi, Manali Singha, Gopal Srivastava, Limeng Pu, J. Ramanujam, and Michal Brylinsky
        Frontiers in Pharmacology: 587

    - MolDQN: https://www.nature.com/articles/s41598-019-47148-x Molecular optimization using deep neural networks
```


<!-- Gameplan -->
1. Enter pocket sequence -> Generate candidates
2. Optimize candidates using mol DQN given a specific set of values eg: logP etc Multireward should be used + GNN's & stop tokens
3. Identitfy potential pockets given a protein pdb file
4. Prediction of binding affinity to target
4. Build a GUI using streamlit / gradio that allows you to do all of the above

Protein -> Pocket -> Candidates -> Optimised candidates + A beautiful GUI to wrap it all up nicely



<!-- MolDQN + Pocket2Drug -->
1. Create a dataset: pocket features + target molecule
2. Embed pocket features + next moleculear action states
3. Select the best action based on the predictions from neural network
4. Run for the length of the molecule (selfies embedding) at which point generate EOS 
5. Reward function:
    - 1. Reward only at the very end based on tanimoto similarity then linearly discount the 


<!-- Things to try -->
Right now updating at the end of each episode
Try updating at the end of every N episodes

Negative pairing model

IE: given true and false molecule pocket pairs the model has to classify which belong together and which don't
Iterativly build up the molecular structure at each point classifying whether 


<!-- Next -->
1. Add substrucutre rewards
2. Get tensorbaord working so you can better monitor progress