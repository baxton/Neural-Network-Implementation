
//
// g++ -O3 -I. -msse3 ann_classifier.cpp -shared -o ann.dll
// g++ -O3 -I. ann_classifier.cpp -shared -o ann2.dll
// g++ -O3 -I. ann_classifier.cpp -shared -o libann.so
//

//
// g++ -O3 -I. -mstackrealign -msse3 ann_classifier.cpp -shared -o ann_sse.dll
// g++ -O3 -I. ann_classifier.cpp -shared -o ann.dll
//

/*
 * Wrapper for Python
 *
 *
 */



#include <cstdio>
#include <vector>
#include <ann.hpp>


typedef double DATATYPE;

extern "C" {

/*
    void* ann_fromfile(const char* fname) {
        FILE* fin = fopen(fname, "rb");
        if (!fin)
            return NULL;

        fseek(fin, 0, SEEK_END);
        size_t size = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        size_t buffer_size = size / sizeof(DATATYPE);
        ma::memory::ptr_vec<DATATYPE> buffer(new DATATYPE[buffer_size]);
        size_t read = fread(buffer.get(), size, 1, fin);

        ma::ann<DATATYPE>* ann = new ma::ann<DATATYPE>(buffer.get());

        return ann;
    }
*/


    void* ann_create(const int* layers, int size, int regres) {
        ma::random::seed();

        std::vector<int> sizes;

        for (int i = 0; i < size; ++i) {
            sizes.push_back(layers[i]);
        }

//        sizes.push_back(45 - 1);
//        sizes.push_back(220);
//        sizes.push_back(220);
//        sizes.push_back(333);
//        sizes.push_back(222);
//        sizes.push_back(111);
//        sizes.push_back(91);
//        sizes.push_back(48);
//        for (int i = 0; i < 100; ++i)
//            sizes.push_back(51);
//        sizes.push_back(1);

        ma::ann_leaner<DATATYPE>* ann = new ma::ann_leaner<DATATYPE>(sizes, regres);
        return ann;
    }


    void ann_fit(void* ann, const DATATYPE* X, const DATATYPE* Y, int rows, DATATYPE* alpha, DATATYPE lambda, int epoches) {

        int cost_cnt = 0;
        DATATYPE prev_cost = 999.;
        DATATYPE cost = 0;

        bool increaced = false;

        for (int e = 0; e < epoches; ++e) {
            cost = static_cast< ma::ann_leaner<DATATYPE>* >(ann)->fit_minibatch(X, Y, rows, *alpha, lambda);

            if (isinf(cost) || isnan(cost) || prev_cost < cost) {
                //if (*alpha > 0.000000000001)
                    *alpha /= 2.;
            }


            if (0 < e && 0 == (e % 200))
                cout << setprecision(16) << cost << " [" << *alpha << "]" << (prev_cost < cost ? ">>>" : "") << endl;

            if (*alpha == 0.) {
                break;
            }

            prev_cost = cost;

            if (prev_cost < cost && 10. >= (cost - prev_cost) && !increaced) {
                increaced = true;
                epoches *= 2;
            }

        }
        cout << setprecision(16) << cost << " [" << *alpha << "]" << endl;
    }


    void ann_free(void* ann) {
        delete static_cast< ma::ann_leaner<DATATYPE>* >(ann);
    }

    void ann_predict(void* ann, const DATATYPE* X, DATATYPE* predictions, int rows) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->predict(X, predictions, rows);
    }


    void ann_save(void* ann) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->save_weights();
    }

    void ann_restore(void* ann) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->restore_weights();
    }

    void ann_shift(void* ann) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->random_shift();
    }

}




