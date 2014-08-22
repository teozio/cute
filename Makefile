CXX=mpicxx-openmpi-gcc49
CXXFLAGS=-O2 -Wall -I/opt/local/include
LIBS=-framework vecLib

PROGRAM=test.x

SRC=Matrix.o \
		main.o

all: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(LIBS) -o $(PROGRAM)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -r *.o *.x

