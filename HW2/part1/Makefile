CC = g++
CCFLAGS = -O3 -std=c++17 -Wall 
LINKOPT = -lpthread

all: pi

pi: pi.cpp
	$(CC) $(CCFLAGS) pi.cpp -o pi.out $(LINKOPT)

clean:
	rm -f *.o *.s