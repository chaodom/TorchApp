#ifndef TORCHAPP_TORCHREGRESS_H
#define TORCHAPP_TORCHREGRESS_H

#include "TorchApp.h"

namespace torchapp {

	class TORCHAPP_API TorchRegress :public TorchApp
	{
	public:
		TorchRegress() {}
		virtual ~TorchRegress() {}

		vector<float> regress(Mat img);
		vector<vector<float>> regress(vector<Mat> imgs);
	};

}

#endif