#ifndef __MDPARGUMENTPARSER__
#define __MDPARGUMENTPARSER__

#include <cstdlib>
#include <unistd.h>
#include <string>

struct MDPArgumentParser {

		MDPArgumentParser(int argc, char ** argv) {

			// Write instructions
			instructions = std::string("Please, execute as follow: ") + argv[0] +
					" -f <instance filename>\n[-r <number of executions>]\n[-p <population size>]\n[-k <Number of contestants>]\n[-c <Probability of crossovering>]\n[-m <Probability of mutation>]\n[-s <Max number of seconds>]\n";

			//Initialize argument with defaults values
			n_executions = 10;
			pop_size = 100;
			n_contestants = 3;
			p_cross = 0.65;
			p_mut = 0.4;
			max_seconds = 60;

			char l;

			while((l=getopt(argc, argv, "f:r:p:k:c:m:s:")) != -1) {

					switch(l) {

							case 'f':
								inst_filename = optarg;
								break;

							case 'r':
								n_executions = atoi(optarg);
								break;

							case 'p':
								pop_size = atoi(optarg);
								break;

							case 'k':
								n_contestants = atoi(optarg);
								break;

							case 'c':
								p_cross = atof(optarg);
								break;

							case 'm':
								p_mut = atof(optarg);
								break;

							case 's':
								max_seconds = atoi(optarg);
								break;

							case '?':
								if (l >= 'c' && l <= 's') {
									throw std::string("Option \"") + l + "\" requires value";
								}
								else {
									throw std::string("Unknow option: ") + l;
								}

							default:
								throw std::string("Error on parsering");

					}

			}

			// Check wheter inst_filename is empty
			if (argc > 1 && inst_filename.empty()) {
				throw "Instance filename not specified";
			}

		}

		std::string inst_filename; //<- Filename of instance data file
		size_t n_executions; //<- Number of executions to perform
		size_t pop_size; // Size of populations
		size_t n_contestants; //Number of contestants to choose in tournament selector
		double p_cross; //Probability of crossovering
		double p_mut; //Probability of mutation
		size_t max_seconds; //Max number of seconds

		std::string instructions;

};

#endif
