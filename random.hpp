

#if !defined RANDOM_DOT_HPP
#define RANDOM_DOT_HPP


#include <cstdlib>
#include <ctime>
#include <limits>



namespace ma {

namespace random {

    void seed(int s=-1) {
        if (s == -1)
            ::srand(time(NULL));
        else
            ::srand(s);
    }

    int randint() {
#if 0x7FFF < RAND_MAX
        return ::rand();
#else
        return (int)::rand() * ((int)RAND_MAX + 1) + (int)::rand();
#endif
    }


    int randint(int low, int high) {
        int r = randint();
        r = r % (high - low) + low;
        return r;
    }

    void randint(int low, int high, int* numbers, int size) {
        for (int i = 0; i < size; ++i) {
            numbers[i] = randint() % (high - low) + low;
        }
    }

    /*
     * Retuns:
     * k indices out of [0-n) range
     * with no repetitions
     */
    void get_k_of_n(int k, int n, int* numbers) {
        for (int i = 0; i < k; ++i) {
            numbers[i] = i;
        }

        for (int i = k; i < n; ++i) {
            int r = randint(0, i);
            if (r < k) {
                numbers[r] = i;
            }
        }
    }

    template<class T>
    void rand(T* buffer, int size) {
        for (int i = 0; i < size; ++i) {
            T tmp = ::rand();
            buffer[i] = tmp / RAND_MAX;
        }
    }


    double generateGaussian(double mu, double sigma)
    {
    	const double epsilon = std::numeric_limits<double>::min();
    	const double two_pi = 2.0*3.14159265358979323846;

    	static double z0, z1;
    	static bool generate = false;
    	generate = !generate;

    	if (!generate)
    	   return z1 * sigma + mu;

    	double u1, u2;
    	do
    	 {
    	   u1 = ::rand() * (1.0 / RAND_MAX);
    	   u2 = ::rand() * (1.0 / RAND_MAX);
    	 }
    	while ( u1 <= epsilon );

    	z0 = ::sqrt(-2.0 * ::log(u1)) * ::cos(two_pi * u2);
    	z1 = ::sqrt(-2.0 * ::log(u1)) * ::sin(two_pi * u2);
    	return z0 * sigma + mu;
    }


    template<class T>
    void randn(T* buffer, int size, double mu, double sigma) {
        for (int i = 0; i < size; ++i) {
            buffer[i] = generateGaussian(mu, sigma);
        }
    }

}

}


#endif
