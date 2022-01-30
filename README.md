# GeneticProgrammingRegression.jl
Project 3 of the STMO course

## Introduction
In this project I will analyze a personal data set, using the package `SymbolicRegression` that takes a genetic programming approach to perform regression.

### Data set

For this data set, I tracked my mood state each day over a period of 644 days. It includes following variables:

**Response variables** (score 0-10)
- frustration
- energy
- clarity
- happiness
- guilt
- emotionality
- anxiety
- confidence

**Predictor variables**
- sleepStart (bed time, time of day with 0 being midnight)
- sleepEnd (waking up, time of day with 0 being midnight)
- sleepDuration (amount slept, hours)
- sleepNap (napped during day, hours)
- meditation (score 0-2)
- exercise (physical exercise, score 0-2)

### Regression method

The `SymbolicRegression` package uses genetic programming to find a symbolic formula that predicts a response variable, using a set of predictor variables. Under the hood, the symbolic equations are represented as trees with nodes corresponding to an operator. The binary operator `+` for example can be linked to a node of a tree with two branches that respresent the two terms of the sum, each of which can be a sub-equation (or subtree) themselves. The leafs of these equation trees are either numbers or variables.

In `EquationSearch`, the central function of the package, these tree-equations are evolved every iteration into similar but better performing equations. A set of promising equations are kept in the *Hall of fame* and are evolved in further iterations.

Lastly, the dominating Pareto frontier is calculated for the equations in the hall of fame. This corresponds to the equations for which all simpler equations perform worse.
