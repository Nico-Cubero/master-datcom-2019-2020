#include "MDPsolution.hpp"
#include "MDPinstance.hpp"
#include <cstdlib>
//#include <limits>

// Implementations of method of MDPSolution

MDPSolution::MDPSolution(const MDPInstance & instance):
		_instance( (const MDPInstance *) & instance) {

	//Allocate memory
	_values = new uint8_t[_instance->getNumObjs()];

	//Initialises the values vector
	this -> reset();
}

MDPSolution::MDPSolution(const MDPSolution & sol) {
	//Copy the values of the solution passed
	_instance 			= sol._instance;
	//_numberParameters 	= sol.getNumberParameters();
	_values  			= new uint8_t[_instance->getNumObjs()];
	_fitness 			= sol._fitness;
	//_updated 			= sol._updated;

	for (size_t i=0; i<_instance->getNumObjs(); i++)
		this->_values[i] = sol._values[i];
}

//Function: getNumObjs
size_t MDPSolution::getNumObjs() const {return _instance->getNumObjs();};

//Function: getSubsetSize
size_t MDPSolution::getSubsetSize() const {return _instance->getSubsetSize();};

//Function: swap
void MDPSolution::swap(MDPSolution & sol, size_t x1, size_t x2) {
	uint8_t aux;

	for (size_t i = x1; i < x2; i++) {
		aux = this->_values[i];
		this->_values[i] = sol._values[i];
		sol._values[i] = aux;
	}
}

void MDPSolution::updateFitness() {
	_fitness = _instance->evaluate(*this);
}

//Function: reset
void MDPSolution::reset() {

	memset(_values, (uint8_t)0, _instance->getNumObjs());

	// Set the non valid solution fitness
	_fitness = 0.0;//std::numeric_limits<double>::lowest();
}

//Function: operator =
const MDPSolution & MDPSolution::operator= (const MDPSolution & sol) {

	for (size_t i=0; i<getNumObjs(); i++)
		this->_values[i] = sol._values[i];

	_fitness = sol._fitness;

	return *this;
}

//Function: operator <<
std::ostream& operator<<(std::ostream & flujo, const MDPSolution & sol) {

	flujo << "[ ";

	for (size_t i=0; i<sol.getNumObjs(); i++) {
		flujo << (int)sol[i]<<" ";
	}

	flujo << "]";


	return flujo;
}

//Function: generateRandomSolution
void MDPSolution::generateRandomSolution(const MDPInstance & instance,
																													MDPSolution & sol) {

	size_t obj;

	sol.reset(); // Reset the solution

	//Generates new solution
	for (size_t i=0; i<instance.getSubsetSize(); i++) {

		// Insert a random object into the subset
		do {
			obj = rand() % instance.getNumObjs();
		} while (sol[obj]);

		sol.put(obj);
	}

	sol.updateFitness();
}
