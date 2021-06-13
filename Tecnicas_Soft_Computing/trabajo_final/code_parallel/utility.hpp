#ifndef __UTILITY_HPP__
#define __UTILITY_HPP__

#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>

//Function: mean
double mean(const std::vector<double> & vector) {

	double avg(0.0);

	//Compute the sum
	for (double x : vector) {
		avg += x;
	}

	//Split between number elements
	avg /= vector.size();

	return avg;
}

//Function: stdDesv
double stdDesv(const std::vector<double> & vector) {

	double std (0.0);
	double avg = mean(vector);

	//Compute the cuadratic sum
	for (double x : vector) {
		std += x*x;
	}

	//Split betwenn the number elements
	std /= vector.size();

	//Deduct the cuadratic average
	std -= avg*avg;

	return sqrt(std);
}

//Function: is_dir
bool is_dir(const std::string & pathname) {

	struct stat info;

	if (stat(pathname.c_str(), &info) != 0) {
		return false;
	}
	else if (info.st_mode & S_IFDIR) {
		return true;
	}
	else {
		return false;
	}
}

#endif
