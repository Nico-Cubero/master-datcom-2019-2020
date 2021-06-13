#ifndef __EVOLUTIONARYALGORITHM_HPP__
#define __EVOLUTIONARYALGORITHM_HPP__

#include <vector>
#include <memory>
//#include <omp.h>
#include "metaheuristic.hpp"
#include "MDPinstance.hpp"
#include "MDPsolution.hpp"
#include "parentSelector.hpp"
#include "crossoverOperator.hpp"
#include "mutationOperator.hpp"
#include "utility.hpp"
//using namespace std;

/**
 * Class which implements a Generational Genetic Algorithm with Elitism for MD
 */
class EvolutionnaryAlgorithm : public Metaheuristic {

	protected:

	// Parameters
	size_t 		_popsize;		// Number of coexisting solutions in population
	std::vector<std::shared_ptr<MDPSolution>> _population;	// Points to survivors of the last replacement
	std::vector<std::shared_ptr<MDPSolution>> _nextgen;	// Points to where the spawning offspring starts
	std::vector<std::shared_ptr<MDPSolution>> _parents;	//Points to parents selected on each generation


	// Operators
	const ParentSelector *	_selector;	// Mechanism to choose which parents will breed
	const CrossoverOperator * _crosser;	// Combines genes of parents into descendants
	const MutationOperator	  * _mutator;	// Induces random variations into chromosomes
	//Instance  * _instance;		// Describes a black box RPO problem

	/* Calculated Info */
	std::shared_ptr<MDPSolution>	_fittest;	// Index of the best solution in pop array
	std::shared_ptr<MDPSolution>	_newbest;	// Index of the best solution in nextgen array
	std::shared_ptr<MDPSolution>	_weakest;	// Index of the worst solution in nextgen array

	std::vector<double> _population_fitness;


	/* Private Methods */
	/* Built-In replacement Operator. Elitism applies after spawning */
	void genReplacement(){
		/* starting indexes */
		size_t i = 0;
		_weakest = _nextgen[0];

		/* Ensuring the survival of the fittest */
		for(; i<_popsize ; i++){

			_nextgen[i]->updateFitness();
			_numEvals++;

			if( *_nextgen[i] > *_fittest) {
				_fittest = _nextgen[i]; 	// All hail the new king!

			/* Store per-evaluation values: current & best */
				break;
			}
			/* Locating a victim */
			if(*_weakest > *_nextgen[i] )
				_weakest = _nextgen[i] ;

			/* Store per-evaluation values: current & best */
			_population_fitness[i] = _nextgen[i]->getFitness();
		}

		if ( i == _popsize ) {
			*_weakest = *_fittest;
			_fittest =  _weakest;
		}

		/* Here continues from the break */
		for(;i<_popsize ; i++){
			_nextgen[i]->updateFitness();

			if(*_nextgen[i]>*_fittest ){
				_fittest = _nextgen[i];
			}

			/* Store per evaluation values: current & best */
			_population_fitness[i] = _nextgen[i]->getFitness();
		}

		_nextgen.swap(_population);
	}

	void init_population() {

		//#pragma omp parallel for
		for (size_t i = 0; i < _population.size(); i++) {

			// Generate new random solutions
			MDPSolution * sol = new MDPSolution(*_instance);
			MDPSolution::generateRandomSolution(*_instance, *sol);

			_population[i].reset(sol);

			/* Tracking best solution */
			if (!_fittest || *_fittest > *sol) {
				_fittest = _population[i];
			}

			//Register fitness of solution
			_population_fitness[i] = _population[i] -> getFitness();
		}

		if (*_fittest > _bestSolution)
			_bestSolution = *_fittest;

	}

	public:

	/* Constructor */
	EvolutionnaryAlgorithm (const MDPInstance & instance,
				  const StopCondition & stopCondition,
					const ParentSelector & parSelector,
					const CrossoverOperator & crossOp,
					const MutationOperator & mutOp,
				  const size_t size
				  ):
				  Metaheuristic(instance, stopCondition), _popsize( size ),
						_selector((const ParentSelector *)&parSelector),
						_crosser((const CrossoverOperator *)&crossOp),
						_mutator((const MutationOperator *)&mutOp)  {

		/* Allocating memory for both current and next generation */
		_population.resize(size);
		_nextgen.resize(size);
		_parents.resize(size);

		_name = "Evolutionnary_algorithm";

	}

	/* Destructor */
	~EvolutionnaryAlgorithm (){
		_population.clear();
		_nextgen.clear();
		_parents.clear();

		_population_fitness.clear();
	}

	void run(void) {

		reset();
		_population_fitness.clear();

		//Allocate memory for saving the results
		_curFitness.resize(_stopCondition.maxIters, 0.0);
		_bestFitness.resize(_stopCondition.maxIters, 0.0);
		_population_fitness.resize(_popsize, 0.0);

		// Generating random solutions and tracking the best one
		init_population();

		/* Storing per-generation best and average fitness */
		_curFitness[_numIters]	= mean(_population_fitness);
		_bestFitness[_numIters] = _bestSolution.getFitness();

		/* Execution of metaheuristic  */
		while ( !stopCondition_is_met() ){


		// Parent Selector
		_selector -> select(_population, _parents);


		/* Crossbreeding */   // crosser is fed by selector
		_crosser -> breed (_parents, _nextgen);



		// Mutation of solutions
		_mutator -> mutate(_nextgen);


		/* Updating fitness and forcing survival of the fittest */
		genReplacement();

		//Keep the best solution
		if (*_fittest>_bestSolution)
			_bestSolution = *_fittest;


		/* Storing per-generation best and average fitness */
		_curFitness[_numIters]	= mean(_population_fitness);
		_bestFitness[_numIters] = _bestSolution.getFitness();

		/* Generation complete! */
		_numIters++;
		}

		_numIters--;
		_elapsedTime = _timer.elapsed_time(Timer::VIRTUAL);

		//Resize the size of arrays of results
		_curFitness.resize(_numIters);
		_bestFitness.resize(_numIters);
	}
};
#endif
