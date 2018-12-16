#ifndef H_NAIVE
#define H_NAIVE

#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <assert.h>
#include <tensorflow/c/c_api.h>
using namespace std;

typedef unsigned int uint32;


//#ifndef CHANGE_PLANAR
//#define CHANGE_PLANAR
//#endif
//
//#ifndef REPLACE_ANG
//#define REPLACE_ANG
//#endif

//#ifndef REPLACE_ALL
//#define REPLACE_ALL
//#endif

//#ifndef HuziDebug2
//#define HuziDebug2
//#endif
//
#ifndef TUNECONTEXT
#define TUNECONTEXT
#endif
// ////
//#ifndef HUZIDEBUG3
//#define HUZIDEBUG3
//#endif
//#ifndef HUZIDEBUG4
//#define HUZIDEBUG4
//#endif

//#define HUZIDEBUG5
// #define ONLY8X8
#define DEEPRDO
// #define FALLTOP

#define EnableDeep

#define NET_CNN 1
#define NET_FC 2

#define CUSTOM_BLOCK_SIZE 256
#define DumpPred
#define PADDING 4
// #define DumpContext

enum Custom_Flags
{
	Context_Ready,
	Context_Not_Ready,
	MSE_Available,
	Context_Saved,
	Context_OK,
	Context_Ready_Half,
	Dump_OK,
	Dump_Not_OK,
	Default
};

template<typename T>
class Hmat
{
public:
	uint32 dim1, dim2;
	T *p;
	Hmat(uint32 d1, uint32 d2) : dim1(d1), dim2(d2)
	{
		p = new T[d1 * d2];
		memset(p, 0, d1 * d2 * sizeof(T));
	}
	~Hmat()
	{
		delete[] p;
	}
	T* operator [] (uint32 it)
	{
		return &(p[it * dim2]);
	}
	Hmat<T> &operator = (Hmat<T> & ref)
	{
		memcpy(p, ref.p, dim1 * dim2 * sizeof(T));
		return *this;
	}
	void clear()
	{
		memset(p, 0, dim1 * dim2 * sizeof(T));
	}
	void dumptxt(string fn, string fmt = "%.5f ")
	{
		FILE *of = fopen(fn.c_str(), "w");
		for (int i = 0; i < dim1; i++)
		{
			for (int j = 0; j < dim2; j++)
			{
				fprintf(of, fmt.c_str(), (*this)[i][j]);
			}
			fprintf(of, "\n");
		}
		fclose(of);
	}
};

typedef Hmat<float> * pHmat;

template<typename T>
void read_dense(string src, Hmat<T>* &k, Hmat<T>* &b, uint32 dims[], string id)
{
	ifstream shape_conf(src + "_shape" + id + ".txt");
	ifstream bias_file(src + "_bias" + id);
	ifstream kernel_file(src + "_kernel" + id);

	int input_len, output_len;
	shape_conf >> input_len >> output_len;
	dims[0] = input_len;
	dims[1] = output_len;
	b = new Hmat<T>(1, output_len);
	for (int i = 0; i < output_len; ++i)
	{
		T tmp;
		bias_file >> tmp;
		(*b)[0][i] = tmp;
	}

	k = new Hmat<T>(input_len, output_len);
	for (int i = 0; i < input_len; ++i)
	{
		for (int j = 0; j < output_len; ++j)
		{
			T tmp;
			kernel_file >> tmp;
			(*k)[i][j] = tmp;
		}
	}
}


template<typename T>
void dot(Hmat<T> *a, Hmat<T> *b, Hmat<T> *dst)
{
	uint32 target_dim1 = a->dim2;
	uint32 target_dim2 = b->dim1;
	assert(a->dim2 == b->dim1);
	Hmat<T> *target = dst;
	for (int i = 0; i < a->dim1; ++i)
	{
		for (int j = 0; j < b->dim2; ++j)
		{
			for (int k = 0; k < a->dim2; ++k)
			{
				(*target)[i][j] += (*a)[i][k] * (*b)[k][j];
			}
		}
	}
}

template<typename T>
void add(Hmat<T> *a, Hmat<T> *b, Hmat<T> *dst)
{
	assert(a->dim1 == b->dim1);
	assert(a->dim2 == b->dim2);
	Hmat<T> *target = dst;
	for (int i = 0; i < a->dim1; ++i)
	{
		for (int j = 0; j < a->dim2; ++j)
		{
			(*target)[i][j] = (*a)[i][j] + (*b)[i][j];
		}
	}
}

template<typename T>
void relu(Hmat<T> *a)
{
	for (int i = 0; i < a->dim1; ++i)
	{
		for (int j = 0; j < a->dim2; ++j)
		{
			(*a)[i][j] = (*a)[i][j] > 0 ? (*a)[i][j] : 0;
		}
	}
}

template<typename T>
void sigmoid(Hmat<T> *a)
{
	for (int i = 0; i < a->dim1; ++i)
	{
		for (int j = 0; j < a->dim2; ++j)
		{
			(*a)[i][j] = 1 / (1 + exp(-(*a)[i][j]));
		}
	}
}

template<typename T>
double mask_mean(Hmat<T> *a)
{
	double sum = 0;
	int cnt = 0;
	for (int i = 0; i < a->dim1; ++i)
	{
		for (int j = 0; j < a->dim2; ++j)
		{
			if ((*a)[i][j] != 0)
			{
				sum += (*a)[i][j];
				cnt++;
			}
		}
	}
	if (cnt == 0) return 0;
	else return sum / cnt;
}


class PredNet
{
public:
	vector<pHmat> network;
	vector<pHmat> placeholder;
	void Init(int input_dim, string id);
	void Clear();
	pHmat predict(pHmat input_mat);
	~PredNet();
};

// class PredictedBlockCache
// {
// public:
// 	Hmat<unsigned short> *blks[5];
// 	int x[5], y[5];
// 	/* b0 b1m0 b1m1 b2m0 b2m1 */
// 	PredictedBlockCache()
// 	{
// 		memset(x, -1, sizeof(int) * 5);
// 		memset(y, -1, sizeof(int) * 5);
// 		for (int i = 0; i < 5; ++i)
// 		{
// 			blks[i] = new Hmat<unsigned short>(CUSTOM_BLOCK_SIZE, CUSTOM_BLOCK_SIZE);
// 		}
// 	}
// 	~PredictedBlockCache()
// 	{
// 		for (int i = 0; i < 5; ++i)
// 		{
// 			delete(blks[i]);
// 		}
// 	}
// 	int read(int rx, int ry, int idx, int m, Hmat<unsigned short> *dst)
// 	{
// 		if (idx == (CUSTOM_BLOCK_SIZE==8)?12:48)
// 		{
// 			return -1;
// 		}
// 		if (idx == 0)
// 		{
// 			if (rx == x[0] && ry == y[0])
// 			{
// 				memcpy(dst->p, blks[0]->p, CUSTOM_BLOCK_SIZE * CUSTOM_BLOCK_SIZE * sizeof(short));
// 			}
// 			else return -1;

// 		}
// 		else //idx 4 or 8
// 		{
// 			int setn = idx / ((CUSTOM_BLOCK_SIZE==8)?4:16) * 2 + m - 1;
// 			if (rx == x[setn] && ry == y[setn] )
// 			{
// 				memcpy(dst->p, blks[setn]->p, CUSTOM_BLOCK_SIZE * CUSTOM_BLOCK_SIZE * sizeof(short));
// 			}
// 			else return -1;
// 		}
// 		return 0;
// 	}
// 	int write(int rx, int ry, int idx, int m, Hmat<unsigned short> *src)
// 	{
// 		if (idx == (CUSTOM_BLOCK_SIZE==8)?12:48)
// 		{
// 			return -1;
// 		}
// 		if (idx == 0)
// 		{
// 			memcpy(blks[0]->p, src->p, CUSTOM_BLOCK_SIZE * CUSTOM_BLOCK_SIZE * sizeof(short));
// 			x[0] = rx; y[0] = ry;
// 		}
// 		else //idx 4 or 8
// 		{
// 			int setn = idx / ((CUSTOM_BLOCK_SIZE==8)?4:16) * 2 + m - 1;
// 			memcpy(blks[setn]->p, src->p, CUSTOM_BLOCK_SIZE * CUSTOM_BLOCK_SIZE * sizeof(short));
// 			x[setn] = rx;
// 			y[setn] = ry;
// 		}
// 		return 0;
// 	}
// };

class Mem
{
public:
	Hmat<unsigned short> *mat_ori;
	Hmat<unsigned short> *mat_prd;
	Hmat<unsigned short> *mat_o;
	vector<Hmat<unsigned short> * > mats;
	vector<Hmat<unsigned short> * > dump_mats;
	vector<Hmat<unsigned short> * > mat_store;
	vector<pair<int, int> > coors;
	vector<double> rdcost;
	int replace;
	Custom_Flags flag, flag2, dump_flag;
	int cnt, cnt2, cnt3;
	int usedPU_Y, totalPU_Y;
	int network_mode;
	double mses[35];
	int deepOn;
	double sum, sel_sum, rdcost_sum, rdcost_sum_sel;
	int currentTagX, currentTagY, cachedX, cachedY;
	// PredictedBlockCache pcache;
	short getPartIdx[128];
	float norm_value;
	Mem(): replace(0), flag(Default), flag2(Default), dump_flag(Dump_Not_OK), cnt(0), cnt2(0), cnt3(0),
		usedPU_Y(0), totalPU_Y(0), network_mode(-1), deepOn(0), sum(0), sel_sum(0), rdcost_sum(0), rdcost_sum_sel(0),
		currentTagX(0), currentTagY(0), cachedX(-1), cachedY(-1), norm_value(0)
	{
		memset(mses, 0, sizeof(mses));
		mat_ori = new Hmat<unsigned short>(2, CUSTOM_BLOCK_SIZE * 2 + 1);
		mat_prd = new Hmat<unsigned short>(CUSTOM_BLOCK_SIZE, CUSTOM_BLOCK_SIZE);
		mat_o = new Hmat<unsigned short>(CUSTOM_BLOCK_SIZE, CUSTOM_BLOCK_SIZE);
		memset(getPartIdx, -1, sizeof(getPartIdx));
		getPartIdx[4] = 1;
		getPartIdx[8] = 4;
		getPartIdx[16] = 16;
		getPartIdx[32] = 64;
		getPartIdx[64] = 256;
	}
	~Mem()
	{
		delete mat_ori;
		delete mat_prd;
		delete mat_o;
		int length = mats.size();
		for (int i = 0; i < length; i++)
		{
			delete mats[i];
		}
	}
	bool check_valid_block_size(int blk_size)
	{
		if (blk_size!=8 && blk_size!=16)
		{
			return false;
		}
		return true;
	}
	void init_context()
	{
		for (int i = 0; i < 5; i++)
		{
			mats.push_back(new Hmat<unsigned short>(CUSTOM_BLOCK_SIZE, CUSTOM_BLOCK_SIZE));
		}
	}
	void push(Hmat<unsigned short> *mat)
	{
		Hmat<unsigned short> *p = new Hmat<unsigned short>(mat->dim1, mat->dim2);
		*p = *mat;
		dump_mats.push_back(p);
	}
	void clear(int num = 5)
	{
		for (int i = 0; i < num; i++)
		{
			dump_mats.pop_back();
		}
	}
	void store(int num = 6, double threshold = 0, int x = 0, int y = 0, double rd = 0.0)
	{
		for (int i = 0; i < num; i++)
		{
			mat_store.push_back(dump_mats[i]);
		}
		for (int i = 0; i < num; i++)
		{
			dump_mats.pop_back();
		}
		coors.push_back(make_pair(x, y));
		rdcost.push_back(rd);
	}
	void dump()
	{
		string prefix = ".\\dumps\\";
		int length = mats.size();
		assert(length < 1000000);
		for (int i = 0; i < length; i++)
		{
			stringstream ss;
			ss << setw(6) << setfill('0') << i;
			mats[i]->dumptxt(prefix + ss.str(), "%d ");
		}
	}
	void dump_total_txt(string fn = "dump_file.txt", string fmt = "%d ")
	{
		FILE *of = fopen(fn.c_str(), "w");
		int length = mats.size();
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < mats[i]->dim1; j++)
			{
				for (int k = 0; k < mats[i]->dim2; k++)
				{
					fprintf(of, fmt.c_str(), (*mats[i])[j][k]);
				}
				fprintf(of, "\n");
			}
		}
		fclose(of);
	}
	void dump_total_txt_selected(string fn = "dump_file.txt", string fmt = "%d ")
	{
		FILE *of = fopen(fn.c_str(), "w");
		int length = mat_store.size();
		for (int i = 0; i < length; i++)
		{
			if (i % 7 == 0)
				fprintf(of, "%d %d %.2f\n", coors[i / 7].first, coors[i / 7].second, rdcost[i / 7]);
			for (int j = 0; j < mat_store[i]->dim1; j++)
			{
				for (int k = 0; k < mat_store[i]->dim2; k++)
				{
					fprintf(of, fmt.c_str(), (*mat_store[i])[j][k]);
				}
				fprintf(of, "\n");
			}
		}
		fclose(of);
	}

    void dump_total_bin_info(string fn = "dump_file.txt", string fmt = "%d ")
	{
		FILE *of = fopen(fn.c_str(), "wb");
		int length = mat_store.size();
		printf("length: %d\n", length);
		for (int i = 0; i < length; i++)
		{
			if (i % 2 == 0)
			{
				int coors_bin[2];
				coors_bin[0] = coors[i / 2].first;
				coors_bin[1] = coors[i / 2].second;
				fwrite(coors_bin, 4, 2, of);
				coors_bin[0] = rdcost[i/2];
				coors_bin[1] = mat_store[i]->dim1;
				fwrite(coors_bin, 4, 2, of);
			}
			// fprintf(of, "%d %d %.2f\n", coors[i/7].first, coors[i/7].second, rdcost[i/7]);
			for (int j = 0; j < mat_store[i]->dim1; j++)
			{
				// for (int k = 0; k < mat_store[i]->dim2; k++)
				// {
				//   // fprintf(of, fmt.c_str(), (*mat_store[i])[j][k]);
				// }
				fwrite((*mat_store[i])[j], 2, mat_store[i]->dim2, of);
				// fprintf(of, "\n");
			}
		}
        int coors_bin[2];
        coors_bin[0] = -1;
        coors_bin[1] = -1;
        fwrite(coors_bin, 4, 2, of);
		fclose(of);
	}
	void dump_total_bin(string fn = "dump_file.txt", string fmt = "%d ")
	{
		FILE *of = fopen(fn.c_str(), "wb");
		int length = mat_store.size();
		printf("length: %d\n", length);
		for (int i = 0; i < length; i++)
		{
			if (i % 2 == 0)
			{
				int coors_bin[2];
				coors_bin[0] = coors[i / 2].first;
				coors_bin[1] = coors[i / 2].second;
				fwrite(coors_bin, 4, 2, of);
			}
			// fprintf(of, "%d %d %.2f\n", coors[i/7].first, coors[i/7].second, rdcost[i/7]);
			for (int j = 0; j < mat_store[i]->dim1; j++)
			{
				// for (int k = 0; k < mat_store[i]->dim2; k++)
				// {
				//   // fprintf(of, fmt.c_str(), (*mat_store[i])[j][k]);
				// }
				fwrite((*mat_store[i])[j], 2, CUSTOM_BLOCK_SIZE, of);
				// fprintf(of, "\n");
			}
		}
		fclose(of);
	}
	void dump_total_txt_skip(string fn = "dump_file.txt", string fmt = "%d ")
	{
		FILE *of = fopen(fn.c_str(), "w");
		int length = mats.size();
		for (int i = 5; i < length; i++)
		{
			for (int j = 0; j < mats[i]->dim1; j++)
			{
				for (int k = 0; k < mats[i]->dim2; k++)
				{
					fprintf(of, fmt.c_str(), (*mats[i])[j][k]);
				}
				fprintf(of, "\n");
			}
		}
		fclose(of);
	}
};

class BaseNetPredict
{
public:
	Hmat<unsigned short> *inbound, *outbound;
	int width;
	virtual int init(string model_file, string trained_file, int _width) = 0;
	virtual int predict() = 0;
};

class TF_Predict:public BaseNetPredict
{
public:
	// Hmat<unsigned short> * inbound, *outbound;
	float* raw_input_data;
	int64_t* raw_input_dims;
	TF_Session* sess;
	TF_Graph* graph;
	TF_Status* status;
	virtual TF_Buffer* read_file(const char* file);
	// virtual void free_buffer(void* data, size_t length);
	// virtual void deallocator(void* ptr, size_t len, void* arg);
	virtual int init(string model_file, string trained_file, int _width);
	virtual int predict();
};

class TF_HalfPredict:public BaseNetPredict
{
public:
	// Hmat<unsigned short> * inbound, *outbound;
	float* raw_input_data;
	int64_t* raw_input_dims;
	TF_Session* sess;
	TF_Graph* graph;
	TF_Status* status;
	virtual TF_Buffer* read_file(const char* file);
	// virtual void free_buffer(void* data, size_t length);
	// virtual void deallocator(void* ptr, size_t len, void* arg);
	virtual int init(string model_file, string trained_file, int _width);
	virtual int predict();
};

class Nets
{
public:
	int num_nets;
	vector<BaseNetPredict*> nets;
	Nets(): num_nets(0) {}
	void append(BaseNetPredict *net)
	{
		num_nets++;
		nets.push_back(net);
	}
};

#endif
