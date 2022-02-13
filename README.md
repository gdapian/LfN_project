# Learning from networks project
Plant-pollinator networks describe the interaction between different types of pollinator and plants of a certain territory. 

**Dataset**: these graphs are publicly available at [https://icon.colorado.edu/#!/networks](https://icon.colorado.edu/#!/networks) has been modified to make them easier to import.

# To do

- [x] Centralities
- [x] Motifs
- [x] Random graph
- [x] p-value
- [x] z-score
- [x] BiRank

# Dependencies
- `numpy`
- `matplotlib`
- `pandas`
- `networkx`
- `os`
- `shutil`
- `birankpy`
- `jupyter notebook`
# Setup
You can clone the repository on your local machine using the following command:
```bash 
git clone https://github.com/gdapian/LfN_project.git
```
# Usage
Go to the directory where the repository is
```bash
cd LfN_project/src
```
Open the notebook
```bash
jupyter notebook Plant_pollinator_project.ipynb
```
# References
Benno I. Simmons et al. “Motifs in bipartite ecological networks: uncovering indirect interactions”. In: (2018).
DOI: https://doi.org/10.1111/oik.05670

Xiangnan He Ming Gao Min-Yen Kan Dingxian Wang. “BiRank: Towards Ranking on Bipartite Graphs”. In:
(2017). DOI: https://arxiv.org/abs/1708.04396

Sebastian Wernicke. “Efficient detection of network motifs”. In: (2006). DOI: 10.1109/TCBB.2006.51
