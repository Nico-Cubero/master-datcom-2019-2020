CPPFLAGS=-W -Wall -Wextra -std=c++11 -g


MDPMetaheuristic: main.cpp Timer.o MDPsolution.o MDPinstance.hpp MDPsolution.hpp metaheuristic.hpp evolutionaryAlgorithm.hpp mutationOperator.hpp crossoverOperator.hpp utility.hpp MDPargumentParser.hpp
	@echo "Compilando el programa principal"
	g++ main.cpp Timer.o MDPsolution.o $(CPPFLAGS) -o $@ -fopenmp

MDPsolution.o: MDPsolution.cpp MDPsolution.hpp
	@echo "Compilando MDPSolution"
	g++ $(CPPFLAGS) -c $<

Timer.o: Timer.cpp Timer.h
	@echo "Compilando Timer.cpp"
	g++ $(CPPFLAGS) -c $<

clean:
	@echo "Borrando todos los código objeto"
	rm -rf *.o
