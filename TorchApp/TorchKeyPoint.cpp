#include "TorchKeyPoint.h"
#include <torch/script.h>

namespace torchapp {

	vector<DetectOutput> TorchYoloPose::detectKeyPoint(Mat img, double scoreThreshold, vector<int> classes)
	{
		return detectKeyPoint(vector<Mat>{img}, scoreThreshold, classes)[0];
	}

	vector<vector<DetectOutput>> TorchYoloPose::detectKeyPoint(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo::detect(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchYolo11Pose::detectKeyPoint(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloPose::detectKeyPoint(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo11Pose::detectKeyPoint(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloPose::detectKeyPoint(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchYolo26Pose::detectKeyPoint(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloPose::detectKeyPoint(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo26Pose::detectKeyPoint(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloPose::detectKeyPoint(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchKeyPointRCNN::detectKeyPoint(Mat img, double scoreThreshold, vector<int> classes)
	{
		ModelOutput result;
		m_scale_w = 1.0;
		m_scale_h = 1.0;
		if (m_bResize)
		{
			Mat img_ = resize(img, m_minSize, m_maxSize);
			result = forward(img_);
			m_scale_w = img.cols * 1.0 / img_.cols;
			m_scale_h = img.rows * 1.0 / img_.rows;
		}
		else
			result = forward(img);
		auto out_tuple = result.toTuple()->elements();
		// out_tupleµÄË³ÐòÊÇ(boxes, classes, keypoint_heatmaps, keypoints, scores)
		auto tensor_clsID = out_tuple[1].toTensor().toType(torch::kInt).to(at::kCPU);
		auto tensor_score = out_tuple[4].toTensor().to(at::kCPU);
		auto tensor_box = out_tuple[0].toTensor().to(at::kCPU);
		auto tensor_keypoint = out_tuple[3].toTensor().to(at::kCPU);
		return process(&tensor_clsID, &tensor_score, &tensor_box, scoreThreshold, classes, nullptr, &tensor_keypoint);
	}

	vector<vector<DetectOutput>> TorchKeyPointRCNN::detectKeyPoint(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		vector<vector<DetectOutput>> results;
		results.reserve(imgs.size());
		for (auto& img : imgs)
			results.push_back(detectKeyPoint(img, scoreThreshold, classes));
		return results;
	}

}