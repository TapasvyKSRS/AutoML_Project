from __future__ import annotations

from enum import IntEnum
from typing import Callable, List, Optional, Tuple

import numpy as np

# If this is your first time seeing @property, please see this example and feel free
# to google more if you need
#
# Class Example:
#
#   def __init__(self):
#       self.count = 0
#       self.word = "hello world"
#
#   @property
#   def myvalue(self) -> str:
#       return self.word + "  " + str(self.count)
#
#   @myvalue.setter(self):
#   def myvalue(self, new_word: str) -> None
#       self.count += 1
#       self.word = new_word
#
# # Notice how no functions calls are made with () and it's treated like an attribute
# example = Example
# print(example.value)  # "hello world 0"
# example.myvalue = "hello mars"
# print(example.value)  # "hello mars 1"


class Recombination(IntEnum):
    """Enum defining the recombination strategy choice."""

    NONE = -1
    UNIFORM = 0
    INTERMEDIATE = 1


class Mutation(IntEnum):
    """Enum defining the mutation strategy choice."""

    NONE = -1
    UNIFORM = 0
    GAUSSIAN = 1


class ParentSelection(IntEnum):
    """Enum defining the parent selection choice."""

    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


class Member:
    """Class to simplify member handling."""

    def __init__(
        self,
        initial_x: np.ndarray,
        target_function: Callable,
        bounds: Tuple[float, float],
        mutation: Mutation,
        recombination: Recombination,
        sigma: Optional[float] = None,
        recom_prob: Optional[float] = None,
        logging: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        initial_x : np.ndarray
            Initial coordinate of the member.
        target_function : Callable
            The target function that determines the fitness value.
        bounds : Tuple[float, float]
            Lower and upper bounds for all coordinates.
        mutation : Mutation
            Mutation strategy.
        recombination : Recombination
            Recombination strategy.
        sigma : Optional[float]
            Standard deviation for Gaussian mutation.
        recom_prob : Optional[float]
            Probability used in uniform recombination.
        logging : bool
            Whether to print debugging information.
        """
        self._x = initial_x.astype(float)
        self._f = target_function
        self._bounds = bounds
        self._mutation = mutation
        self._recombination = recombination
        self._sigma = sigma
        self._recom_prob = recom_prob

        self._age = 0
        self._x_changed = True
        self._fit = 0.0

        self.logging = logging

    @property
    def fitness(self) -> float:
        """
        Retrieve the fitness, recalculating it if x changed.

        Note: For this exercise, lower fitness values are better.
        """
        if self._x_changed:
            self._x_changed = False
            self._fit = self._f(self._x)

        return self._fit

    @property
    def x_coordinate(self) -> np.ndarray:
        """The current x coordinate."""
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value: np.ndarray) -> None:
        """Set the new x coordinate."""
        lower, upper = self._bounds
        assert np.all((lower <= value) & (value <= upper)), f"Member out of bounds, {value}"
        self._x_changed = True
        self._x = value

    def mutate(self) -> Member:
        """Mutation which creates a new offspring.

        As a side effect, it increments the age of this member.

        Returns
        -------
        Member
            The mutated member created from this member.
        """
        new_x = self.x_coordinate.copy()

        if self._mutation == Mutation.UNIFORM:
            new_x += np.random.uniform(-1, 1, size=new_x.shape)

        elif self._mutation == Mutation.GAUSSIAN:
            if self._sigma is None:
                raise ValueError("Sigma has to be set when gaussian mutation is used")
            new_x += np.random.normal(0, self._sigma, size=new_x.shape)

        elif self._mutation == Mutation.NONE:
            new_x = self.x_coordinate.copy()

        else:
            raise RuntimeError(f"Unknown mutation {self._mutation}")

        lower, upper = self._bounds
        new_x = np.clip(new_x, lower, upper)

        child = Member(
            new_x,
            self._f,
            self._bounds,
            self._mutation,
            self._recombination,
            self._sigma,
            self._recom_prob,
            logging=self.logging,
        )

        self._age += 1
        return child

    def recombine(self, partner: Member) -> Member:
        """Recombination of this member with a partner.

        Parameters
        ----------
        partner : Member
            The other member to combine with.

        Returns
        -------
        Member
            A new member based on the combination of this member and the partner.
        """
        if self._recombination == Recombination.INTERMEDIATE:
            new_x = (self.x_coordinate + partner.x_coordinate) / 2

        elif self._recombination == Recombination.UNIFORM:
            if self._recom_prob is None:
                raise ValueError("For uniform recombination, recombination probability must be given")

            mask = np.random.rand(*self.x_coordinate.shape) < self._recom_prob
            new_x = np.where(mask, self.x_coordinate, partner.x_coordinate)

        elif self._recombination == Recombination.NONE:
            new_x = self.x_coordinate.copy()

        else:
            raise RuntimeError(f"Unknown recombination {self._recombination}")

        lower, upper = self._bounds
        new_x = np.clip(new_x, lower, upper)

        child = Member(
            new_x,
            self._f,
            self._bounds,
            self._mutation,
            self._recombination,
            self._sigma,
            self._recom_prob,
            logging=self.logging,
        )

        self._age += 1
        return child

    def __str__(self) -> str:
        """Makes the class easily printable."""
        return f"Population member: Age={self._age}, x={self.x_coordinate}, f(x)={self.fitness}"

    def __repr__(self) -> str:
        """Makes the class printable if it is an entry in a list."""
        return self.__str__() + "\n"


class EA:
    """A class implementing evolutionary algorithm strategies."""

    def __init__(
        self,
        target_func: Callable,
        population_size: int = 10,
        problem_dim: int = 2,
        problem_bounds: Tuple[float, float] = (-30, 30),
        mutation_type: Mutation = Mutation.UNIFORM,
        recombination_type: Recombination = Recombination.INTERMEDIATE,
        selection_type: ParentSelection = ParentSelection.NEUTRAL,
        sigma: float = 1.0,
        recom_proba: float = 0.5,
        total_number_of_function_evaluations: int = 200,
        children_per_step: int = 5,
        fraction_mutation: float = 0.5,
        logging: bool = False,
    ):
        """
        Parameters
        ----------
        target_func : Callable
            Callable target function we optimize.
        population_size : int
            Total population size.
        problem_dim : int
            Dimension of each member's x coordinate.
        problem_bounds : Tuple[float, float]
            Lower and upper bounds for all coordinates.
        mutation_type : Mutation
            Mutation strategy.
        recombination_type : Recombination
            Recombination strategy.
        selection_type : ParentSelection
            Parent selection strategy.
        sigma : float
            Standard deviation for Gaussian mutation.
        recom_proba : float
            Probability used in uniform recombination.
        total_number_of_function_evaluations : int
            Maximum allowed function evaluations.
        children_per_step : int
            Number of children to produce per step.
        fraction_mutation : float
            Probability of mutation instead of recombination.
        logging : bool
            Whether to print debugging information.
        """
        assert 0 <= fraction_mutation <= 1
        assert 0 < children_per_step
        assert 0 < total_number_of_function_evaluations
        assert 0 < sigma
        assert 0 < problem_dim
        assert 0 < population_size

        self.logging = logging

        self.population = [
            Member(
                np.random.uniform(*problem_bounds, problem_dim),
                target_func,
                problem_bounds,
                mutation_type,
                recombination_type,
                sigma,
                recom_proba,
                logging=self.logging,
            )
            for _ in range(population_size)
        ]
        self.population.sort(key=lambda member: member.fitness)

        self.pop_size = population_size
        self.selection = selection_type
        self.max_func_evals = total_number_of_function_evaluations
        self.num_children = children_per_step
        self.frac_mutants = fraction_mutation
        self._func_evals = population_size

        self.trajectory = [self.population[0]]

        self.all_members = []
        self.fitness_values = []

        if self.logging:
            print(f"Average fitness of population: {self.get_average_fitness()}")

    def get_average_fitness(self) -> float:
        """The average fitness of the current population."""
        return np.mean([member.fitness for member in self.population])

    def select_parents(self) -> List[int]:
        """Select one parent.

        Returns
        -------
        List[int]
            A list containing one selected parent index.
        """
        population_size = len(self.population)

        if self.selection == ParentSelection.NEUTRAL:
            parent_ids = [int(np.random.choice(population_size))]

        elif self.selection == ParentSelection.FITNESS:
            fitness_values = np.array([member.fitness for member in self.population])

            # We minimize the objective, so lower fitness should get higher probability.
            weights = np.max(fitness_values) - fitness_values

            if np.sum(weights) == 0:
                probabilities = np.ones(population_size) / population_size
            else:
                probabilities = weights / np.sum(weights)

            parent_ids = [int(np.random.choice(population_size, p=probabilities))]

        elif self.selection == ParentSelection.TOURNAMENT:
            tournament_size = min(3, population_size)

            tournament_indices = np.random.choice(
                population_size,
                size=tournament_size,
                replace=False,
            )

            tournament_fitness = [
                self.population[idx].fitness for idx in tournament_indices
            ]

            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parent_ids = [int(winner_idx)]

        else:
            raise RuntimeError(f"Unknown parent selection {self.selection}")

        return parent_ids

    def step(self) -> float:
        """Performs one step of the algorithm.

        Steps:
        1. Parent selection
        2. Offspring creation
        3. Survival selection

        Returns
        -------
        float
            The average population fitness.
        """
        parent_ids: List[int] = []
        for _ in range(self.num_children):
            parent_ids.extend(self.select_parents())

        children: List[Member] = []

        for parent_id in parent_ids:
            parent = self.population[parent_id]

            if np.random.rand() < self.frac_mutants:
                child = parent.mutate()
            else:
                possible_partners = [idx for idx in parent_ids if idx != parent_id]

                if len(possible_partners) == 0:
                    child = parent.mutate()
                else:
                    partner_id = int(np.random.choice(possible_partners))
                    partner = self.population[partner_id]
                    child = parent.recombine(partner)

            children.append(child)
            self._func_evals += 1

        self.all_members.extend(children)
        self.fitness_values.extend([child.fitness for child in children])

        self.population.extend(children)
        self.population.sort(key=lambda member: member.fitness)
        self.population = self.population[: self.pop_size]

        self.trajectory.append(self.population[0])

        return self.get_average_fitness()

    def optimize(self) -> Member:
        """The optimization loop performing the desired number of function evaluations.

        Returns
        -------
        Member
            The best member of the population after optimization.
        """
        step = 1

        while self._func_evals < self.max_func_evals:
            avg_fitness = self.step()
            best_fitness = self.population[0].fitness

            lines = [
                "=========",
                f"Step: {step}",
                "=========",
                f"Avg. fitness: {avg_fitness:.7f}",
                f"Best. fitness: {best_fitness:.7f}",
                f"Func evals: {self._func_evals}",
                "----------------------------------",
            ]

            if self.logging:
                print("\n".join(lines))

            step += 1

        return self.population[0]


if __name__ == "__main__":
    from src.target_function import ackley

    np.random.seed(0)

    dimensionality = 2
    max_func_evals = 500 * dimensionality
    pop_size = 20

    for selection in ParentSelection:
        ea = EA(
            target_func=ackley,
            population_size=pop_size,
            problem_dim=dimensionality,
            selection_type=selection,
            total_number_of_function_evaluations=max_func_evals,
        )
        optimum = ea.optimize()

        print(optimum)
        print("#" * 120)
