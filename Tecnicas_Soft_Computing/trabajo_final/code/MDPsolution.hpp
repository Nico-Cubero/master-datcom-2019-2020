#ifndef __MDPSOLUTION_HPP__
#define __MDPSOLUTION_HPP__

//#include "MDPinstance.hpp"
//#include <cstdlib>
//#include <cmath>
#include <iostream>
#include <cstdlib>

class MDPInstance;

/*Class: MDPSolution
 *Description: Represents a candidate solution for the MD Problem
*/
class MDPSolution {


	uint8_t * _values; //<- Boolean vector indicating wheter a object is included in subset or not

	const MDPInstance * _instance;			//<- MDPInstance of the problem used
	double 			 _fitness;			//<- The fitness value of the solution
	size_t _n_objs;							//<- Total number of objects included in the solution


	public:

		//Constructor
		MDPSolution(const MDPInstance & instance);


		MDPSolution(const MDPSolution & sol);

		//Observers

		/*Function: getNumObjs
		 *Description: Returns the number of objects considered for the problem
		 */
		inline size_t getNumObjs() const;

		/*Function: getNumObjs
		 *Description: Returns the size of subset objects considered for the problem
		 */
		inline size_t getSubsetSize() const;

		/*Function: getNumObjsInc
		 *Description: Returns the actual number of objects included in this solution
		*/
		inline size_t getNumObjsInc() const {return _n_objs;};

		/*Function: is_included
		  *Description: Returns wheter object obj is included in the solution or not
		*/
		inline uint8_t is_included(size_t obj) const {return _values[obj];};

		/*Function: getFitness
		*/
		inline double getFitness() const {return _fitness;};

		//Modifiers

		/*Function: put
		* Description: Put an object into the subset (set binary flag to 1 for this object)
		*/
		inline void put(size_t obj) {if (_values[obj] != 1) {(this->_n_objs)++; _values[obj] = (uint8_t)1;}};

		/*Function: pull
		* Description: Pull an object from the subset (set binary flag to 0 for this object)
		*/
		inline void pull(size_t obj) {if (_values[obj] != 0) {(this->_n_objs)--; _values[obj] = (uint8_t)0;}};

		/*Function: swap_objs
		 *Description: Swap the insertion of two objects into the subset
		*/
		inline void swap_objs(size_t obj1, size_t obj2) {
			uint8_t aux = this->_values[obj1];
			this->_values[obj1] = this->_values[obj2];
			this->_values[obj2] = aux;
		}

		/*Function: swap
		 *Description: Echange the compounds of two solutions existing between two
		 *	indexes provided.
		*/
		void swap(MDPSolution & sol, size_t x1, size_t x2);


		/*Function: updateFitnessValue
		 *Description: update the fitness value of the solution
		*/
		inline void updateFitness();

		/*Function: reset
		 *Description: Function which clears the content of the parameter
		*/
		void reset();

		/*Function: rebuild
		 *Description: Put or pull random objects into the subset until getting the
		 *	the subset size set by the instance problem
		*/
		void rebuild();

		//Operators

		/*Function: operator []
		 *Description: Let to know wheter an object is included in the subset or not
		*/
		inline uint8_t operator[] (size_t obj) const {return _values[obj];};

		/*Function: operator -
		 *Description: Function used for coparing the fitness value of the solutions
		*/
		inline double operator- (const MDPSolution & sol) const {
			return this->getFitness() - sol.getFitness();
		}

			//Overload
		inline double operator- (double fitness) const {
			return this->getFitness() - fitness;
		}

			//overload
		friend inline double operator-(double fitness, const MDPSolution & sol) {
			return fitness - sol.getFitness();
		}

		/*Function: operator >
		 *Description: Function used for computing the difference of fitness value between two solutions
		*/
		inline bool operator> (const MDPSolution & sol) const {
			return this->getFitness() > sol.getFitness();
		}

			//overload
		inline bool operator> (double fitness) const {
			return this->getFitness() > fitness;
		}

		/*Function: operator <
		 *Description: Function used for comparing the fitness value of  two solutions
		*/
		inline bool operator< (const MDPSolution & sol) const {
			return this->getFitness() < sol.getFitness();
		}

		//overload
		inline bool operator< (double fitness) const {
			return this->getFitness() < fitness;
		}

		/*Function: operator =
		 *Description: Function which copy the value of solution parameters into another one
		*/
		const MDPSolution & operator= (const MDPSolution & sol);

		/*Operator <<
		 *Description: Function used for printing solutions on the screen.
		*/
		friend std::ostream& operator<<(std::ostream & flujo, const MDPSolution & sol);

		//Static functions

		/*Function: generateRandomMDPSolution
		 *Description: Function used for generating a random solution
		*/
		static void generateRandomSolution(const MDPInstance & instance,
																														MDPSolution & sol);


		//Destructor
		~MDPSolution() {
			delete [] _values;
			_values=NULL;
			_instance=NULL;
		}
};

#endif
