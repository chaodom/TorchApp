#ifndef TORCHAPP_TORCHFINAL_H
#define TORCHAPP_TORCHFINAL_H

#include "TorchClassify.h"
#include "TorchRegress.h"
#include "TorchCRNN.h"
#include "TorchSegment.h"
#include "TorchKeypoint.h"

namespace torchapp {

	class TORCHAPP_API TorchYolo11Final final :public TorchYolo11Seg, public TorchYolo11Pose
	{
	public:
		TorchYolo11Final() {}
		~TorchYolo11Final() {}

		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	};


	class TORCHAPP_API TorchYolo26Final final :public TorchYolo26Seg, public TorchYolo26Pose
	{
	public:
		TorchYolo26Final() {}
		~TorchYolo26Final() {}

		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	};


	class TORCHAPP_API TorchRCNNFinal final :public TorchMaskRCNN, public TorchKeyPointRCNN 
	{
	public:
		TorchRCNNFinal() {}
		~TorchRCNNFinal() {}

		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	};

}
#endif#pragma once
