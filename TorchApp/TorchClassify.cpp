#include "TorchClassify.h"
#include <torch/script.h>

namespace torchapp {

	vector<ClassifyOutput> TorchClassify::classify(Mat img, int N)
	{
		return classify(vector<Mat>{ img }, N)[0];
	}

	vector<vector<ClassifyOutput>> TorchClassify::classify(vector<Mat> imgs, int N)
	{
		auto result = forward(imgs).toTensor();	// [batch_size, class_num]
		// 打印
		//print(result[0]);
		N = MIN(N, result.size(-1));
		auto softmaxs = result.softmax(-1);
		auto results = softmaxs.sort(-1, true);
		auto [probs, indices] = softmaxs.sort(-1, true);
		// 仅当tensor不在CPU时才拷贝，减少冗余拷贝
		torch::Device cpu_device(torch::kCPU);
		if (indices.device() != cpu_device)
		{
			indices = indices.to(cpu_device);
			probs = probs.to(cpu_device);
		}
		// 确保内存连续（避免stride导致的指针访问异常）
		indices = indices.contiguous();
		probs = probs.contiguous();
		vector<vector<ClassifyOutput>> outputs;
		outputs.reserve(result.size(0));
		for (int i = 0; i != result.size(0); ++i)
		{
			vector<ClassifyOutput> output;
			output.reserve(N);
			for (int n = 0; n < N; ++n)
			{
				ClassifyOutput co;
				co.cls_id = indices[i][n].item<int>();
				co.cls_name = co.cls_id < m_vecClsNames.size() ? m_vecClsNames[co.cls_id] : std::to_string(co.cls_id);
				co.score = probs[i][n].item<float>();
				output.push_back(co);
			}
			outputs.push_back(output);
		}
		return outputs;
	}
}