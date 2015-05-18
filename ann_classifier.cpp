
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

        ma::ann_leaner<DATATYPE>* ann = new ma::ann_leaner<DATATYPE>(sizes, regres);
        return ann;
    }


    void ann_fit(void* ann, const DATATYPE* X, const DATATYPE* Y, int rows, DATATYPE* alpha, DATATYPE lambda, int epoches) {

        int cost_cnt = 0;
        DATATYPE prev_cost = 99999999.;
        DATATYPE cost = 0;

        bool increaced = false;

        for (int e = 0; e < epoches; ++e) {
            cost = static_cast< ma::ann_leaner<DATATYPE>* >(ann)->fit_minibatch(X, Y, rows, *alpha, lambda);

            if (isinf(cost) || isnan(cost) || prev_cost < cost) {
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


    void ann_get_weights(void* ann, DATATYPE* bb, DATATYPE* ww) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->get_bb(bb);
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->get_ww(ww);
    }

    void ann_set_weights(void* ann, DATATYPE* bb, DATATYPE* ww) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->set_bb(bb);
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->set_ww(ww);
    }

    void ann_get_output(void* ann, DATATYPE* Y, int l) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->get_output(Y, l);
    }


    void ann_set_output_scale(void* ann, double val) {
        static_cast< ma::ann_leaner<DATATYPE>* >(ann)->set_output_scale(val);
    }

}




