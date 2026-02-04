#include "TorchCRNN.h"
#include <torch/script.h>

namespace torchapp {

	const string TorchCRNN::s_strSpaceChar = "-";

	CRNNOutput TorchCRNN::decode(vector<string> elements, vector<double> probs)
	{
		string text;
		vector<double> scores;
		int pre_index = 0;
		if (elements.size() > 0 && elements[0] != s_strSpaceChar)
			text += elements[0];
		for (size_t elementIndex = 1; elementIndex < elements.size(); ++elementIndex)
			if (elements[elementIndex] != s_strSpaceChar && elements[elementIndex - 1] != elements[elementIndex])
			{
				text += elements[elementIndex];
				if (text.size() > 1)
				{
					double score = *std::max_element(probs.begin() + pre_index, probs.begin() + elementIndex);
					scores.push_back(score);
					pre_index = elementIndex;
				}
			}
		double score = *std::max_element(probs.begin() + pre_index, probs.end());
		scores.push_back(score);
		CRNNOutput tf = { text,scores };
		return tf;
	}

	CRNNOutput TorchCRNN::recognize(Mat img)
	{
		auto result = forward(img).toTensor();	// [width, batch_size, class_num]
		auto softmaxs = result.softmax(-1);
		auto results = softmaxs.sort(-1, true);
		auto indexs = std::get<1>(results);
		auto scores = std::get<0>(results);
		vector<string> elements;
		vector<double> probs;
		for (int n = 0; n != result.size(0); ++n)
		{
			int idx = indexs[n][0][0].item<int>();
			if (idx == 0)
			{
				elements.push_back(s_strSpaceChar);
				probs.push_back(0.0);
			}
			else
			{
				elements.push_back(m_vecClsNames[idx - 1]);
				probs.push_back(scores[n][0][0].item<float>());
			}
		}
		auto info = decode(elements, probs);
		return info;
	}

}