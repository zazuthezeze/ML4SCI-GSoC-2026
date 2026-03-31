# ML4SCI GSoC 2026 — Genie Project Tasks

## Overview
This repository contains the completed tasks for the ML4SCI Google Summer of Code 2026 application for the Genie projects. Each task is documented in a Jupyter notebook which can be found in the notebooks folder of the respective task directory. The notebooks contain the full explanation of the approach, results, analysis and conclusions for each task.

## Common Task 1 — Autoencoder of Quark/Gluon Events

Goal:
Train an autoencoder to learn compressed representations of quark and gluon jet events across three detector channels (ECAL, HCAL and Tracks) and demonstrate side by side comparisons of original and reconstructed events.

Summary:
A Variational Autoencoder was trained on 139,306 jet events. Four model configurations were tested by varying the latent dimension size and learning rate scheduler.

## Common Task 2 — Jets as Graphs

Goal:
Convert jet images into point cloud graphs and train a graph neural network to classify quark and gluon jets. Then evaluate performance using ROC-AUC at the end.

Summary:
Jet images were converted to graphs by removing all zero pixels and turning each remaining active pixel into a node with position and energy features. Nodes were connected to their 8 nearest neighbours using KNN. An EdgeConv GNN was then trained on these graphs to classify quark and gluon jets. Two models were compared, one trained on 20,000 jets and one on the full 139,306 jets. 

## Specific Task 1 — Deep Graph Anomaly Detection with Contrastive Learning

Goal:
Classify quark and gluon jets using a model that learns data representations through contrastive loss and evaluate classification performance on a test dataset.

Summary:
A contrastive encoder was trained to push similar jets together and different jets apart without ever seeing the class labels. A lightweight classifier was then trained on top of the frozen encoder representations. Six model configurations were tested varying latent dimension size, weight decay and dropout. 