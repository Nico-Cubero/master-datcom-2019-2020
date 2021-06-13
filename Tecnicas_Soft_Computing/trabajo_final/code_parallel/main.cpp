#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <exception>
#include <sys/stat.h>
#include "MDPsolution.hpp"
#include "MDPinstance.hpp"
#include "metaheuristic.hpp"
#include "evolutionaryAlgorithm.hpp"
#include "parentSelector.hpp"
#include "crossoverOperator.hpp"
#include "mutationOperator.hpp"
#include "utility.hpp"
#include "MDPargumentParser.hpp"
#include <limits>

//Prototipes
void save_evolution_results(const Metaheuristic & m, const std::string & filename);
void save_best_results(const std::vector<MDPSolution> & sols, const std::string & filename);
////////////////////////////////////////////////////////////

#define MAX_SECONS_PER_RUN 180
#define MAX_SOLUTIONS_PER_RUN 700000
#define MAX_NUM_EVALUATIONS 700000




//Main function of the program
int main(int argc, char ** argv) {

	try {

	std::string inst_filename;
	std::string out_dirname;
	size_t n_executions;
	size_t pop_size;

	std::vector<unsigned int> seeds = {
		12345678, 23456781, 34567812, 45678123, 56781234, 67812345, 78123456, 81234567,
    12435678, 24356781, 43567812, 35678124, 56781243, 67812435, 78124356, 81243567,
    18435672, 84356721, 43567218, 35672184, 56721843, 67218435, 72184356, 21843567,
    18437652, 84376521, 43765218, 37652184, 76521843, 65218437, 52184376, 21843765,
    18473652, 84736521, 47365218, 73652184, 36521847, 65218473, 52184736, 21847365,
    15473682, 54736821, 47368215, 73682154, 36821547, 68215473, 82154736, 21547368,
    15472683, 54726831, 47268315, 72683154, 26831547, 68315472, 83154726, 31547268,
    65472183, 54721836, 47218365, 72183654, 21836547, 18365472, 83654721, 36547218,
    35472186, 54721863, 47218635, 72186354, 21863547, 18635472, 86354721, 63547218,
    35427186, 54271863, 42718635, 27186354, 71863542, 18635427, 86354271, 63542718,
    36427185, 64271853, 42718536, 27185364, 71853642, 18536427, 85364271, 53642718,
    36428175, 64281753, 42817536, 28175364, 81753642, 17536428, 75364281, 53642817,
    36528174, 65281743, 52817436, 28174365};

	MDPArgumentParser arg_parser(argc, argv);

	//Input data
	if (argc == 1) {
		std::cerr << arg_parser.instructions;
		return -1;
	}
	else {
		//Take filename of instance data file
		inst_filename = arg_parser.inst_filename;

		//Generate output directory filename
		size_t dot_pos = inst_filename.find_last_of('.');

		if (dot_pos != std::string::npos) {
			out_dirname = inst_filename.substr(0, dot_pos) + "_results";
		}
		else {
			out_dirname = inst_filename + "_results";
		}

		n_executions = arg_parser.n_executions;

		if (n_executions >= seeds.size()) {
			std::cout << "Please set a number of executions between 1 and " <<
									seeds.size()-1<<std::endl;
			return -1;
		}

		pop_size = arg_parser.pop_size; // 20 normalmente

		if (pop_size <= 0) {
			std::cout << "Size of population must be greater than 0";
			return -1;
		}

		if (arg_parser.n_contestants <= 0) {
			std::cout << "Number of contestants must be greater then 0";
			return -1;
		}

		if (arg_parser.p_cross < 0 || arg_parser.p_cross > 1) {
			std::cout << "Probability of crossovering must be in [0,1]";
			return -1;
		}

		if (arg_parser.p_mut < 0 || arg_parser.p_mut > 1) {
			std::cout << "Probability of mutation must be in [0,1]";
			return -1;
		}

		if (arg_parser.max_seconds <= 0) {
			std::cout << "Maximum number of executions must be greater than 0";
			return -1;
		}
	}

	//Create or check the existence of output directory
	if (is_dir(out_dirname) == false) {
		std::cout << "Creating directory: \"" << out_dirname << "\"" << std::endl;

		if (mkdir(out_dirname.c_str(), 0777) == -1) {
			std::cerr << "Failed on create directory \""<< out_dirname <<"\"" << std::endl;
			return -1;
		}
	}

	//Defining the problem instance
	MDPInstance instance(inst_filename);

	//Defining the stop condition
	StopCondition stopCondition(MAX_NUM_EVALUATIONS, MAX_SOLUTIONS_PER_RUN,
															arg_parser.max_seconds);

	//Defining the Evolutionary Algorithm
	TournamentSelector selector(arg_parser.n_contestants, pop_size);	// Selector Operator
	CrossoverOperator crosser(instance, arg_parser.p_cross);	// Crossover Operator
	MutationOperator mutator(instance, arg_parser.p_mut);	// Mutation Operator

	//The metaheuristic
	EvolutionnaryAlgorithm evolutionary(instance, stopCondition, selector,
														crosser, mutator, pop_size);

	//For storing the result with the best fitness value of each metaheuristic
	std::vector<MDPSolution>  bestSolutions;


	std::cout << "Running \""<< evolutionary.getName() << "\"" << std::endl;

	//Executions of metaheuristics
	for (size_t i=0; i<n_executions; i++) {

		std::cout << "Running " << i+1 << " with seed " << seeds[i] << std::endl;

		//Running the metaheuristic
		srand(seeds[i]);
		evolutionary.run();

		std::cout << "Execution completed - Time taken: " <<
										evolutionary.elapsed_time() << " s" <<
										" - Iterations performed: " << evolutionary.getNumIters() <<
										" - Evaluations performed: " << evolutionary.getNumEvals() <<
										std::endl;
		std::cout << "Saving Results." << std::endl;

		bestSolutions.push_back(evolutionary.getBestSolution());

		//Save results
		save_evolution_results(evolutionary,
														out_dirname + '/' + "progression_seed=" +
														std::to_string(seeds[i]) + ".csv");
	}

	//Save statiscals of the best results
	save_best_results(bestSolutions, out_dirname + '/' + "summary_results.txt");

	return 0;

	}
	catch(std::exception & e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	catch(const std::string & e) {
		std::cerr << e << std::endl;
		return -1;
	}
	catch(const char * e) {
		std::cerr << e << std::endl;
		return -1;
	}
}

/*Function: save_evolution_results
 *Description: Saves the evolution of current and best fitness values
 *	generated on the execution of metaheuristic into a file
*/
void save_evolution_results(const Metaheuristic & m,
														const std::string & filename) {

	std::ofstream f(filename.c_str());

	if (!f) {
		throw "Failed to open \"" + filename + "\"";
	}

	//Write header
	f << "current_fitness,best_fitness" << std::endl;

	// Write evolution
	for(size_t i = 0; i < m.getNumIters(); i++) {
		f << m.getCurFitness_at_iter(i) << ',' << m.getBestFitness_at_iter(i) << std::endl;
	}

	f.close();
}

/*Function: save_best_results
 *Description: Saves statiscals of the best results
*/
void save_best_results(const std::vector<MDPSolution> & sols,
																	const std::string & filename) {

		// Save best results into a vector
		std::vector<double> results;

		for (const MDPSolution & s : sols) {
			results.push_back(s.getFitness());
		}

		//Compute statical values
		double min = *std::min_element(results.begin(), results.end());
		double max = *std::max_element(results.begin(), results.end());
		double avg = mean(results);
		double std = stdDesv(results);

		std::ofstream f(filename.c_str());

		if (!f) {
			throw "Failed to open \"" + filename + "\"";
		}

		for(size_t i = 0; i < sols.size(); i++) {
			f << "Ejecución nº " << i << ":\n- Best Solution: "<< sols[i] <<
					"\n- Fitness: " << sols[i].getFitness() << std::endl;
		}

		f << "----------------------------------------" << std::endl;
		f << "Max: " << max << ", Min: " << min <<
					", Average: " << avg << ", Standard Desviation: " << std << std::endl;

		f.close();
}

