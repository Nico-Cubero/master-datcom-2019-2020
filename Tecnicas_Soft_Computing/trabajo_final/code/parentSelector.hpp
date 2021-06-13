/******************************************************************************
* File: parentSelector.hpp
* Author: Nicolás Cubero
* Description: Implementation of a parent Selector utility for Evolutionary
*				Algorithm applied for the Maximum Diversity Problem based on a Tournament
*				Selector.
* Date: 7/6/2020
*
* - Máster en Ciencia de Datos e Ingeniería de Computadores -
* - Técnicas de Soft Computing para Aprendizaje y optimización.
*		Redes Neuronales y Metaheurísticas, programación evolutiva y bioinspirada
*******************************************************************************/

#ifndef __PARENTSELECTOR_HPP__
#define __PARENTSELECTOR_HPP__

#include <vector>
#include <memory>
#include "MDPsolution.hpp"

class ParentSelector {

	public:

	/* Returns a fit progenitor */
	virtual void select(const std::vector<std::shared_ptr<MDPSolution>> & pool,
												std::vector<std::shared_ptr<MDPSolution>> & parents) const = 0;
};

class TournamentSelector : public ParentSelector {

	protected:

		size_t _k;				// Number of contestants
		size_t _par_size; // Number of parents to be extracted

		// Selects by index the best candidate of each k tournament
		size_t make_tournament(const std::vector<std::shared_ptr<MDPSolution>> & pool) const {

			size_t best = rand() % pool.size();
			size_t contestant;

			// Performs k tournaments
			for (size_t i = 0; i < _k; i++){

				// Choose one contestant and compare with the best register
				contestant = rand() % pool.size();

				// The Best contestant stands
				if ( *pool[contestant] > *pool[best] )
					 best = contestant;

			}

			return best;
		}

	public:

		/* Constructor */
		TournamentSelector(size_t k, size_t par_size): _k(k), _par_size(par_size){};

		void select(const std::vector<std::shared_ptr<MDPSolution>> & pool,
									std::vector<std::shared_ptr<MDPSolution>> & parents) const {


			//Allocate for par_size parents
			parents.resize(this->_par_size);

			size_t parent;

			//#pragma omp parallel for
			for(size_t i = 0; i < this->_par_size; i++) {
				// Select one parent and save into the parents set
				parent = this->make_tournament(pool);
				parents[i] = pool[parent];
			}

			return ;
		}

};

#endif
