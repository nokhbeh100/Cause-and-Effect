# Cause and Effect
Cause and Effect: Concept Based Explanation of Neural Networks

[article link](https://arxiv.org/abs/2105.07033)


## Download

For downloading the datasets run the scripts:

```
    ./script/dlbroden.sh
    ./script/dlplaces.sh
    ./script/dlzoo.sh
```

## Requirements

* Python Environments

```
    pip3 install numpy matplotlib torch torchvision tqdm progressbar2 pandas opencv-python
```
and pull submodules:
```
    git pull --recurse-submodules
```


## Run Cause and Effect
* For running main analysis run:
```
    python3 forgetting.py
```


* For running deepdream based on the trained concepts run:
```
    python3 deepdream.py
```

* For evaluation of scatter plots and finding the most necessary concepts for a classifier run:
```
    python3 scatter3.py
```


## Reference
If you find the codes useful, please cite this paper
```
@article{zaeem2021cause,
  title={Cause and Effect: Concept-based Explanation of Neural Networks},
  author={Zaeem, Mohammad Nokhbeh and Komeili, Majid},
  journal={arXiv preprint arXiv:2105.07033},
  year={2021}
}
```
