/******************************************************************************
* File: crossoverOperator.hpp
* Author: Nicolás Cubero
* Description: Implementation of a binary crossover operator utility for
*				Evolutionary Algorithm applied for the Maximum Diversity Problem
* Date: 7/6/2020
*
* - Máster en Ciencia de Datos e Ingeniería de Computadores -
* - Técnicas de Soft Computing para Aprendizaje y optimización.
*		Redes Neuronales y Metaheurísticas, programación evolutiva y bioinspirada
*******************************************************************************/

#ifndef __CROSSOVEROPERATOR_HPP__
#define __CROSSOVEROPERATOR_HPP__

#include "MDPinstance.hpp"
#include "MDPsolution.hpp"
#include <omp.h>
//#include "parentSelector.hpp"

class CrossoverOperator{

	protected:

	/* Parameters */
	const MDPInstance  * _instance;			// Problem data
	const double _prob; 							// Probability of crossovering

	/* Function: cross
	*  Description: Perform binary crossover between two chromosomes (solutions)
	*/
	static void cross(MDPSolution & p1, MDPSolution & p2) {

		size_t x;

		x = rand() % p1.getNumObjs();					//First exchanged object

		p1.swap(p2, x, p1.getNumObjs());

		return;
	}

	public:

	/* Constructor */
	CrossoverOperator(MDPInstance & instance, double prob):
			_instance (&instance), _prob(prob) {};


	/* Method to generate offspring in gen2 the same size of gen1 */
	void breed(const std::vector<std::shared_ptr<MDPSolution>> & pop,
							std::vector<std::shared_ptr<MDPSolution>> & nextgen) const {
//Cambiar por stl

		nextgen.resize(pop.size());

		#pragma omp parallel for
		for(size_t i = 0; i < pop.size(); i+=2){

			size_t par1, par2; // The two crossovered patents

			//Select another candidate
			do {
				par1 = rand() % pop.size();
				par2 = rand() % pop.size();
			} while(par2 == par1);


			if (((double)rand()) / RAND_MAX < _prob) {

				MDPSolution * aux1, * aux2;

				// Copy the solutions to form new solutions
				aux1 = new MDPSolution(*pop[par1]);
				aux2 = new MDPSolution(*pop[par2]);

				//Perform crossover between the two parents
				CrossoverOperator::cross(*aux1, *aux2);

				nextgen[i].reset(aux1);
				nextgen[i+1].reset(aux2);
			}
			else {
				nextgen[i] = pop[par1];
				nextgen[i+1] = pop[par2];
			}

		}
	}

};


#endif
