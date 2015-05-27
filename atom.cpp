
#include <iostream>


using namespace std;


void inc(unsigned int& val)
{
    asm volatile ("lock; incl %0"
                    :"=m" (val)
                    :"m" (val)
                );
}


void dec(unsigned int& val)
{
    asm volatile ("lock; decl %0"
                    :"=m" (val)
                    :"m" (val)
                );
}





unsigned int cmpxch(unsigned int& val, unsigned int cmp, unsigned int new_val)
{
    unsigned int prev;

    asm volatile ("lock; cmpxchgl %1, %2"
                    : "=a" (prev)
                    : "r" (new_val), "m" (val), "0"(cmp)
                );
    return prev;
}

bool cas(unsigned int cmp, unsigned int& val1, unsigned int& val2)
{
    unsigned short result;

    asm volatile (
        "lock; cmpxchg %3, %1\n"
        "sete %b0\n"
        : "=r"(result), "+m"(val1), "+a"(cmp)
        : "r"(val2)
        : "memory", "cc"
        );

    return (bool) result;
}



int main(int argc, const char** argv) {

    unsigned int x = 0, y = 1;
    //inc(x);
    cmpxch(x, 0, 5);
    //cas(0, x, y);

    cout << x << " (" << y << ")" << endl;

    return 0;
}
