# choose compiler: icc or g++
CXX = icc
CXXFLAGS += -O2 -std=c++11  -I${COMMON_DIR}
#CXXFLAGS += -O2 -std=c++11

LDFLAGS += -L${COMMON_DIR}
# lib path
LIBS = -lcomn
# source
SRC = $(wildcard *.cpp)
# source object
SRC_OBJ = $(SRC:.cpp=.o)
# source binary
SRC_BIN = main.out

$(SRC_BIN): $(SRC_OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm *o *.out
