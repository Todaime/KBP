# KBP Evaluation

## Introduction

This repository contains the  implementation of the evaluation method for KBP end-to-end systems.

A script for the evaluation of custom methods on the DWIE dataset is also provided.


## Leaderboard
<div align="center">
<table>
  <col>
  <colgroup span="2"></colgroup>
  <colgroup span="2"></colgroup>
  <tr>
    <td rowspan="2">Model</td>
    <th colspan="2" scope="colgroup">Cold-Start</th>
    <th colspan="2" scope="colgroup">Warm-Start</th>
  </tr>
  <tr>
    <th scope="col">F1-micro</th>
    <th scope="col">F1-macro</th>
    <th scope="col">F1-micro</th>
    <th scope="col">F1-macro</th>
  </tr>
  <tr>
    <th scope="row">Elrond</th>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <th scope="row">Elrond+Merit</th>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
</table>
    <caption align="center"> Scores on the DWIE dataset</caption>
</div>
Please contact us so that we can add your results!

## How to use the benchmarker
1. First you will need to download our version of the DWIE dataset through this link , unzip it in the "data/dataset/dwie" directory.
2. You will need to produce with your model a base corresponding to each step of each sequence of the evaluation process. The sequences of text are saved in the "data/dataset/dwie/sequences_for_evaluation.pickle" file.
3. Run the evaluate_system.py.


## References

Link to the source DWIE dataset : https://github.com/klimzaporojets/DWIE
