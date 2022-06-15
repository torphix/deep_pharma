<!-- ADMET Predictors given molecules -->
### Description
1. Base graph neural network model to extract features -> Set2Set (standardize inputs) -> Multiple output head classifiers
2. Having a single backbone coupled with several heads should encourage a diverse array of feature extraction to occur
3. Implement the Various Graph Algorithms from scratch

<!-- MOLDQN iterative refinement of molecules -->

<!-- Visualiser -->
1. Input molecule and get the various proeprty predictions back
2. Input molecule and output several optimized molecules

<!-- Knowledge graph linking molecules, targets, tissues & disease -->
1. Use large language models to extract knowledge graph information from bio medical text


<!-- Important Caveats -->
1. 3D coordinates computation is not deterministic


<!-- Molecule Featurizer -->
``` \
vocab = MolecularVocab()
graph_1 = smiles_to_graph('CCCC', vocab.atom_stoi)
nodes_1, edge_atrs_1, adj_matrix_1 = numerate_features(graph_1)
nodes = normalise_features([nodes_1, nodes_2])
edges = normalise_features([edge_atrs_1, edge_atrs_2])
```

<!-- Project scope -->
1. Graph attention neural network from scratch
2. ADMET multiheaded prediction

<!-- My process -->
1. Define the problem and research SOTA & approaches
2. Read blog posts and watch videos on the topic this usually helps to give a high level overview of the field and background knowledge requriements as well as highlight important areas
3. Read commonly refrenced papers and foundational papers in the field
4. Dig in deeper reading newer reseach + reviews on the field
5. Explore various libraries and API's
6. Establish a quick and dirty baseline for the task
7. Iterate

<!-- References -->
1. Data: https://tdcommons.ai/


<!-- Pick up point -->
1. Construct the training loop for the property prediction
    - Should be in plain pytorch (not lightning or geometric) 
    - Implement the Graph conv network
    - Implement multi head training
    - Should train on multiple property prediction using different network heads and a core backbone
        - 
    - Should be able to test different architechtures and benchmark them against an xgboost model
    - Implement the graph neural networks (the best performing networks that you benchmarked on) from scratch in pytorch
    - Generate visulaiser for generated molecules


1. Input training set, if boolean or regression, and if to use log scale as target

<!-- Benchmarking -->
1. Task: Lipophilicity_AstraZeneca
2. Parameters:
    - Epochs: 20
3. Results:
    - Graph attention networks: 1.5M paramters: 1.11 loss
    - Graph conv networks: 1.5M paramters: 0.952 loss
    - Sage network: 2.8M parameters: loss=0.984
    - Gated Graph Conv: 13M paramters:  loss=0.715
    - Gated Graph Conv: 2.5M paramters: loss=0.602
    - Gated Graph Conv: 1.2M paramters: loss=0.568
    - Res Graph Conv: 1.5M parameters: loss=0.577
    - TransformerConv: 1.5M paramters: loss=0.87
    - XGBoost model
    - 

4. Cluster the classified molecules tother




1. Train code backbone + head model

<!-- Workera -->
1. Test
2. Which answers your unsure on
3. Study 
4. Iterate

<!-- Workera Deep learning quiz -->


<!-- ADME prediction with GNNS -->
<!-- RL for molecular optimisation for desired properties -->
<!-- Underestand each element of theory through and through -->

<!-- Process -->
1. Set up good evaulation pipelines and visualisations of where model is going wrong as well as data distriubtions
2. Set up UI for ADMET
3. Set up MOLDQN interface with the reward function a combination of ADMET prediction ie: Guided optimisation using RL
4. See if grouping tasks together improves performance eg: All toxicity tasks togther all metabolism tasks togther etc
5. Workera questions
6. Power point:
    - Clear explanantion of Graph neural nets and chosen architechture
    - Clear explanation of RL algorithm used

<!-- Add weighted sampler -->
<!-- Finish UI -->
<!-- Power point presentation -->

<!-- Problem -->
1. When you retrain on false positives what tends to happen is you get less false positives but more False negatives (as biases the model in the opposite direction)
2. Therefore for the task eg: toxicitiy prediction its important to ascertain which metric is better to oppimise for ie: okay to let a few toxic molecules through? or better to stop a non toxic molecules


<!-- Resgated conv notes -->
1. Why? 
    - Residuals allow for deeper stacking of layers without loss in accuracy
    - 


<!-- Command -->
 /home/j/miniconda3/envs/deep_pharma/bin/python main.py train -fte=False --finetune_iters=1