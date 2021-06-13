#ifndef __MDPINSTANCE_HPP__
#define __MDPINSTANCE_HPP__

#include <string>
#include <fstream>
#include <limits>
#include <cstring>
#include <cstdlib>
#include "MDPsolution.hpp"

#define LOWEST_REAL_VALUE std::numeric_limits<double>::lowest()

//class MDPSolution;

/*Class: MDPInstance
 *Description. Abstract definition of an instance problem used in the MDP
 * problem consisting on finding a solution which maximizes
 *			de diversity of a subset of objects for which there're defined
 * pair-wise dinstances between the objects.
*/
class MDPInstance {

	private:
		size_t _N;	//<- Size of the original set of objects
		size_t _M;	//<- Size of the subjects

		double ** _d;	//<- The distance between each pair of object
		//double (*_f) (const Solution & sol);	//The function used for the evaluation

	public:


		//Constructors

		// Construct new empty instance
		MDPInstance(size_t N, size_t M): _N(N), _M(M) {

			// Allocate memory for the pairwise distance matrix
			_d = new double* [_N];

			for(size_t i=0; i<_N; i++) {
				_d[i] = new double [_N];

				// Set 0 value to each row
				memset(_d[i], 0.0, _N);
			}

		}

		//Construct instance from the data of a file
		inline MDPInstance(std::string filename) {

			std::ifstream f (filename.c_str());

			if (!f) {
				throw std::string("Failed to load file: \'")+filename+"\'";
			}

			//Read the number of objects and the subset size
			f >> this->_N >> this->_M;

			// Allocate memory for the pairwise distance matrix
			this->_d = new double* [_N];

			for(size_t i=0; i<_N; i++) {
				_d[i] = new double [_N];

				// Set 0 value to each row
				memset(_d[i], 0.0, _N);
			}

			//Read the pairwise distance matrix
			size_t i, j;
			double dist_val;

			while (!f.eof()) {
				f >> i >> j >> dist_val;
				this->_d[i][j] = dist_val;
			}

			//Close the file
			f.close();


		}


		//Observers

		/*Function: getNumObjs
		 *Description: Returns the number of objects considered for the problem
		 */
		inline size_t getNumObjs() const {return _N;};

		/*Function: getNumObjs
		 *Description: Returns the size of subset objects
		 */
		inline size_t getSubsetSize() const {return _M;};

		/*Function: getDist
		 *Description: Returns the distance between two objects
		 */
		inline double getDist(size_t i, size_t j) const {return _d[i][j];};

		/*Function: evaluate
		 *Description: Function which evaluates the fitness value of a proposed solution (the value resulting of evaluating the solution)
		 */
		double evaluate(const MDPSolution & sol) const {

			/* If the number of countabilized object in the solution is greater than
				the subset size, the solution must not be valid so the lowest computable
				real value is returned, in other case, the fitness value is returned
			*/
			if (sol.getNumObjsInc() != _M) return 0.0;

			double fitness (0.0);
			//size_t n_objs (0);

			for (size_t i=0; i < _N; i++) {

				if (!sol[i]) continue;

				for (size_t j=i+1; j < _N; j++) {
					fitness += (sol[i]&sol[j]) * this->getDist(i,j);
				}
			}


			//return n_objs == _M? fitness: 0.0;
			return fitness;
		}


		 //Destructor
		~MDPInstance() {

			//Deallocate pairwise distance matriz memory
			for (size_t i=0; i<_N; i++)
				delete [] _d[i];

			delete [] _d;

		}
};
#endif
