# Decision Tree Learning on the MONK Datasets

## Description  
This project implements decision tree learning using a top-down greedy algorithm, evaluated on the MONK datasets (MONK-1, MONK-2, and MONK-3). These are synthetic classification tasks with known underlying rules, designed to illustrate concepts like entropy, information gain, overfitting, and pruning. The implementation builds trees from scratch and evaluates them both graphically and quantitatively.

## Key Results

| Dataset | Train Accuracy (Unpruned) | Test Accuracy (Unpruned) | Train Accuracy (Pruned) | Test Accuracy (Pruned) | Observations                                 |
|---------|----------------------------|---------------------------|--------------------------|-------------------------|-----------------------------------------------|
| MONK-1  | 100.0%                     | 82.9%                     | 94.6%                    | 88.9%                   | Pruning significantly improved test accuracy        |
| MONK-2  | 100.0%                     | 69.2%                     | 61.4%                    | 67.1%                   | Pruning reduced overfitting but didn’t help test accuracy |
| MONK-3  | 100.0%                     | 94.4%                     | 98.6%                    | 95.4%                   | Already strong generalization, pruning helped marginally  |

- All unpruned trees perfectly fit the training data.
- Pruning helped reduce overfitting and improved generalization in MONK-1 and MONK-3.
- MONK-2 remained a difficult case due to complexity in the class structure.

## Features

- Custom implementation of entropy and information gain
- Top-down greedy decision tree construction based on maximizing information gain
- Evaluation using train/test accuracy
- Visualization of trees using PyQt5
- Pruning through subtree replacement for reduced variance
- Systematic analysis of training/validation data splits to optimize pruning effectiveness

## Dataset

The MONK datasets are synthetic binary classification problems over discrete attributes:
- **Attributes**: 6 (all categorical)
- **Classes**: Binary (0 or 1)
- **Sets**: MONK-1, MONK-2, MONK-3, each with separate training and test subsets

These datasets are defined in `monkdata.py` and loaded as Python objects (`monk1`, `monk1test`, etc.).

## File Structure

- `notebook.ipynb`:  
  Main notebook containing the step-by-step development, from entropy computation to tree construction, evaluation, and pruning.

- `dtree.py`:  
  Contains all logic for entropy, information gain, decision tree creation (`buildTree`), evaluation (`check`), and pruning (`allPruned`).

- `monkdata.py`:  
  Holds the dataset definitions and attribute metadata (attributes, values, labels).

- `drawtree.py`:  
  Uses PyQt5 to display graphical tree representations via the `drawTree()` function.

## Methodology

- Trees are built recursively by selecting the attribute with the highest information gain at each node.
- Leaf nodes are assigned based on the majority class when further splitting is not beneficial or max depth is reached.
- The decision tree is evaluated on unseen data using the `check()` function.
- Pruning is implemented by recursively testing alternatives that replace subtrees with majority-class leaves.

## Installation

```bash
pip install numpy PyQt5
