[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/px3OQngL)
# NAS
## Assignment Instructions / TODOs
In this exercise, you will implement and compare different **Neural Architecture Search (NAS)** techniques. The tasks are divided into multiple parts, ranging from blackbox optimization on a tabular benchmark to differentiable and one-shot NAS methods.

Later in the exercise, you'll apply **DARTS** to discover optimal CNN architecture on the MNIST dataset, learning how to implement **one-shot optimizers** and analyze their results.

Please refer to the [PDF](https://drive.google.com/file/d/1grzJD3HERXHYhxvA51Sge2nDoCjNUHZ8/view?usp=drive_link) for full assignment instructions. 


#### Dependency setup

```bash
pip install -r requirements.txt
```

### Feedback
Please give us feedback by filling out feedback.md file.

### Grading
The grading takes place in the following manner: 
- Autograding via Github Actions (This gives a brief idea of how well your solution performed).
- Respective Teaching Staff evaluates the solutions, and provide necessary feedback on the submitted solutions.

The final assignment grade would be the one decided after evaluation by the teaching staff associated with the exercises.
Overall, the assignments focus on comprehensive understanding of the lecture. 

#### Extra
These assignments were tested with `Python 3.10` and should work for any greater or equal version. You can check your python version by using `Python -V`.
It is highly advised to use some form of virtual environment when working on these assignments, to prevent conflicts between packages for different projects on your computer.
We recommend using Conda for this, but please feel free to use any method that works for you.

We also provide a `Makefile` which has some handy commands for you if you are using the commandline while doing this assignment, run `make help` to find them!

To run a `Makefile` on Windows, among other solutions, you can install `make` via Chocolatey as shown [here](https://chocolatey.org/install).

#### References 
> For details about the search space and encoding, refer to:  
> [*NAS-Bench-101: Towards Reproducible Neural Architecture Search*](https://arxiv.org/abs/2001.00326)
#### Deadline
The assignment is due on **CEST 23:59 of June 19, 2026**.

In case of any issues, please reach out to us via discord classroom forum using the tag "NAS".
