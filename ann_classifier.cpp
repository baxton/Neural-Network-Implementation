
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


    void* ann_create() {
        ma::random::seed();

        std::vector<int> sizes;
        sizes.push_back(78);
        sizes.push_back(100);
        //sizes.push_back(30);
        //sizes.push_back(51);
        //sizes.push_back(51);
        //for (int i = 0; i < 10; ++i)
        //    sizes.push_back(51);
        sizes.push_back(1);

        ma::ann_leaner<DATATYPE>* ann = new ma::ann_leaner<DATATYPE>(sizes);
        return ann;
    }


    void ann_fit(void* ann, const DATATYPE* X, const DATATYPE* Y, int rows, DATATYPE* alpha, DATATYPE lambda, int epoches) {

        int cost_cnt = 0;
        DATATYPE prev_cost = 999.;
        DATATYPE cost = 0;

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
        }
        cout << setprecision(16) << cost << " [" << *alpha << "]" << endl;
    }


    void ann_free(void* ann) {
        delete static_cast< ma::ann_leaner<DATATYPE>* >(ann);
    }

    void ann_predict(void* ann, const DATATYPE* X, DATATYPE* predictions, int rows) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->predict(X, predictions, rows);
    }



}



