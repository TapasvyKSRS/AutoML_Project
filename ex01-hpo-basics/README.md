[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ZCyE1-3w)
# Introduction and Basics of Hyperparameter Optimization

## Assignment Instructions / TODOs
From the lecture, you have learned about evolutionary algorithms and simple methods of hyperparameter optimization.

For part 1, you will be editing `evolution.py`.

For part 2, you will be editing `hpo.py`.

For part 3, you will be editing `neps_hpo` and `plot_trajectories.py`.

For bonus part 4, you will be editing `hpo_suite.py` 

Please refer to the [PDF](https://drive.google.com/file/d/1Q7RvKZ--x-kRvD3agtinSZjUD5ku6Efo/view?usp=sharing) for full assignment instructions. 

To run any individual file without running the tests, from root directory run `python -m src.<module_name>` eg: `python -m src.hpo` 

### Feedback
Please give us feedback by filling out feedback.md file.

### Grading
The grading takes place in the following manner: 
- Autograding via Github Actions (This gives a brief idea of how well your solution performed).
- Respective Teaching Staff evaluates the solutions, and provides necessary feedback on the submitted solutions.

The final assignment grade would be the one decided after evaluation by the teaching staff associated with the exercises.
Overall, the assignments focus on comprehensive understanding of the lecture. 

#### Extra
These assignments were tested with `Python 3.10` and should work for any greater or equal version. You can check your python version by using `Python -V`.
It is highly advised to use some form of virtual environment when working on these assignments, to prevent conflicts between packages for different projects on your computer.
We recommend using Conda for this, but please feel free to use any method that works for you.

We also provide a `Makefile` which has some handy commands for you if you are using the commandline while doing this assignment, run `make help` to find them!

To run a `Makefile` on Windows, among other solutions, you can install `make` via Chocolatey as shown [here](https://chocolatey.org/install).

### 📦 Additional Build Dependencies

If you're installing `hposuite` with, make sure `swig` is installed in your system:

- **macOS**: `brew install swig`
- **Ubuntu**: `sudo apt install swig`
- **Windows**: `choco install swig`
- **Manually**: Use [SWIG installer](https://www.swig.org) and add it to your PATH.
- You can also install swig in your **Conda Environment** using `conda install -c anaconda swig`.

#### Deadline
The assignment is due on **CEST 23:59 of May 01, 2026**.

Incase of any issues, please reach out to us via discord classroom forum using the tag "HPO BASICS".
