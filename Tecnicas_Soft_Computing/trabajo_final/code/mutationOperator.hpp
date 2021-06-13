/******************************************************************************
* File: mutationOperator.hpp
* Author: Nicolás Cubero Torres
* Description: Implementation of a binary mutation operator utility for Evolutionary
*				Algorithm applied for the Maximum Diversity Problem.
* Date: 7/6/2020
*
* - Máster en Ciencia de Datos e Ingeniería de Computadores -
* - Técnicas de Soft Computing para Aprendizaje y optimización.
*		Redes Neuronales y Metaheurísticas, programación evolutiva y bioinspirada
*******************************************************************************/

#ifndef __MUTATIONOPERATOR_HPP__
#define __MUTATIONOPERATOR_HPP__

#include "MDPinstance.hpp"
#include "MDPsolution.hpp"
//#include <omp.h>

//#define MPART 128

class MutationOperator {

	protected:
		const MDPInstance * _instance;
		double _prob;		 // Probability of mutation

	public:

	MutationOperator(const MDPInstance & i, double prob):
		_instance((const MDPInstance *) &i), _prob(prob) {};

	void mutate(std::vector<std::shared_ptr<MDPSolution>> & pop) const{

		size_t obj2;

		//#pragma omp parallel for
		for(size_t i = 0; i < pop.size(); i++) {

			for (size_t obj1 = 0; obj1 < pop[i]->getNumObjs(); obj1++) {

				if (((double)rand()) / RAND_MAX < _prob) {

					// Select the second object to be mutated
					obj2 = rand() % (pop[i]->getNumObjs());

					pop[i]->swap_objs(obj1, obj2);
				}

			}

		}

	}

};


#endif
