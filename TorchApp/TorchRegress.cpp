#include "TorchRegress.h"
#include <torch/script.h>

namespace torchapp {

	vector<float> TorchRegress::regress(Mat img)
	{
		return regress(vector<Mat>{ img })[0];
	}

	vector<vector<float>> TorchRegress::regress(vector<Mat> imgs)
	{
		vector<vector<float>> results;
		auto result = forward(imgs).toTensor();	// [batch_size, output_num]
		result = result.to(at::kCPU);
		for (int i = 0; i != result.size(0); ++i)
		{
			vector<float> res;
			res.assign(result[i].data_ptr<float>(), result[i].data_ptr<float>() + result[i].numel());
			results.push_back(res);
		}
		return results;
	}
}