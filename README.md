# KBP Evaluation

## Introduction

This repository contains the implementation of the evaluation method for end-to-end Knowledge Base Population (KBP) systems.

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
    <th scope="row">Elrond+Merit</th>
    <td>82.5</td>
    <td>75.0</td>
    <td>81.1</td>
    <td>71.1</td>
  </tr>
  <tr>
    <th scope="row">Elrond</th>
    <td>82.0</td>
    <td>74.8</td>
    <td>80.0</td>
    <td>70.2</td>
  </tr>
</table>
    <caption align="center"> Scores on the DWIE dataset</caption>
</div>

If you are interested in the KBP task, do not hesitate to contact us so that we can share your results on the leaderboard or to discuss any idea you might have to build the optimal KBP system!

## How to use the benchmarker
Here is how to proceed to directly evaluate your solution on the DWIE dataset with our benchmarker.

- You can find a sequence file `data/dataset/dwie/sequences_for_evaluation.pickle` that gives the order in which to process the texts in the test set to build the database for ten different orderings.
- After every step of each sequence, you need to produce a file that represents the Knowledge Base built from the information in the previous texts. In Warm-start setting, an initial KB is supplied. Each KB file must be named by its position in the sequence and stored in a folder with the name of the sequence number.
- A KB file must be a pickle file storing a nested dictionnary and structured as follows :

```
{
    "texts": {filename: list[URI of entities extracted in the text]},
    "entities": {
        "relations": [(predicate, (mentions of the object in the file), filename)],
        "attributes": [("type", type_tag, filename) or ("mention", mention, filename)]
        }
}
```


## References

- Link to the source DWIE dataset : https://github.com/klimzaporojets/DWIE
