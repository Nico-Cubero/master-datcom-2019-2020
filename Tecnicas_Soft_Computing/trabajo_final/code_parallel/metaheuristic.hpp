#ifndef __METAHEURISTIC_HPP__
#define __METAHEURISTIC_HPP__

#include <vector>
#include <string>
#include "MDPinstance.hpp"
#include "MDPsolution.hpp"
#include "Timer.h"

/*Struct StopCondition
 *Description: structure used for knowing when the stop condition is met
*/
struct StopCondition {

	// Parametric constructor
	StopCondition(unsigned maxEvals,
				  unsigned maxIters,
				  double   maxTime):
						   maxEvals(maxEvals),
						   maxIters(maxIters),
						   maxTime(maxTime) {};

	// Copy constructor
	StopCondition(const StopCondition & stopCondition):
						   maxEvals(stopCondition.maxEvals),
						   maxIters(stopCondition.maxIters),
						   maxTime(stopCondition.maxTime)  {};

	unsigned maxEvals;
	unsigned maxIters;
	double   maxTime;

};

/*Class Metaheuristic
Desription: Abstract class which represents a general metaheuristic*/
class Metaheuristic {

	protected:

		const MDPInstance * _instance;		 //<- The instance problem

		StopCondition _stopCondition;	 //<- The stop condition
		unsigned _numIters;		 //<- The number of iteration performed
		unsigned _numEvals;		 //<- The number of evaluations performed
		double _elapsedTime;	 //<- Time (in seconds) taken for the execution
		Timer _timer;					 //<- Timer used for controlling the time

		std::string _name; 				 //<- Name of the metaheuristic
		MDPSolution _bestSolution;			//<-The best solution found by the metaheuristic

		// Store the evolution of fitness value
		std::vector<double> _curFitness;
		std::vector<double> _bestFitness;

		//Observers
		/*Function: stopConditionIsMet
		 *Description: Function which checks whether the stop condition is met.
		*/
		bool stopCondition_is_met() {

			double elapsedTime = _timer.elapsed_time(Timer::VIRTUAL);

			if (_stopCondition.maxEvals > 0 &&
				_numEvals >=_stopCondition.maxEvals)
				return true;

			if (_stopCondition.maxIters > 0 &&
				_numIters >= _stopCondition.maxIters)
				return true;

			if (_stopCondition.maxTime > 0 &&
				elapsedTime >= _stopCondition.maxTime)
				return true;

			return false;
		 }

		/*Function: reset
		 *Description: Function used for reseting the state (the counter variables)
		 *	of metaheuristic and make it ready for a new run
		*/
		virtual inline void reset() {
			_numIters=0;
			_numEvals=0;
			_elapsedTime=0.0;
			_timer.reset();

			_bestSolution.reset();
		}


	public:

		//Constructor
		Metaheuristic(const MDPInstance & instance, const StopCondition & stopCondition):
			_instance((const MDPInstance*)&instance), _stopCondition(stopCondition),
			_numIters(0),_numEvals(0), _elapsedTime(0.0), _bestSolution(instance)
			{_timer.reset();};

		//Observers

		/*Function: getName
		 *Description: Function used for knowing the name of metaheuristic
		*/
		inline const std::string & getName() const {return _name;};

		/*Function: getBestSolution
		 *Description: Function used for getting the best solution
		*/
		inline const MDPSolution & getBestSolution() const {return _bestSolution;};

		/*Function: getCurFitness_at_iter
		 *Description: Get the current fitness value got at iteration i
		*/
		inline double getCurFitness_at_iter(size_t i) const {return _curFitness.at(i);};

		/*Function: getBestFitness_at_iter
		 *Description: Get the best fitness value got at iteration i
		*/
		inline double getBestFitness_at_iter(size_t i) const {return _bestFitness.at(i);};

		/*Function: getNumIters
		 *Description: Returns the number of iterations performed
		*/
		inline unsigned getNumIters() const {return _numIters;};

		/*Function: getNumEvals
		 *Description: Returns the number of evaluations performed
		*/
		inline unsigned getNumEvals() const {return _numEvals;};

		/*Function: elapsed_time
		 *Description: Returns the elapsed time taken
		*/
		inline double elapsed_time() const {return _elapsedTime;};

		/*Function: getResults
		 *Description: Function used for copying STL vector on solutions
		*/
		virtual inline void getResults(std::vector<double> & curFitness,
																			std::vector<double> & bestFitness) const {
			curFitness = _curFitness;
			bestFitness = _curFitness;
		}

		/*Function: run
		 *Description: Function used for running the complete execution and to obtain the results
		*/
		virtual void run()  {};


};

#endif
