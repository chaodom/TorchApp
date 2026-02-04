#ifndef TORCHAPP_TORCHCRNN_H
#define TORCHAPP_TORCHCRNN_H

#include "TorchApp.h"

namespace torchapp {

	typedef struct tagCRNNOutput
	{
		string text;
		vector<double> scores;
	}CRNNOutput, * PCRNNOutput;


	class TORCHAPP_API TorchCRNN :public TorchApp
	{
	public:
		TorchCRNN() {}
		virtual ~TorchCRNN() {}

		CRNNOutput recognize(Mat img);

	private:
		CRNNOutput decode(vector<string> elements, vector<double> probs);	// 将输出转为原生字符串

	private:
		static const string s_strSpaceChar;

	};

}

#endif