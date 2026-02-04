#ifndef TORCHAPP_TORCHKEYPOINT_H
#define TORCHAPP_TORCHKEYPOINT_H

#include "TorchDetect.h"

namespace torchapp {

	class TORCHAPP_API TorchKeyPoint :virtual public TorchDetect
	{
	public:
		TorchKeyPoint() {}
		virtual ~TorchKeyPoint() {}

		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {}) = 0;
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {}) = 0;

	};


	class TORCHAPP_API TorchYoloPose :public TorchKeyPoint, virtual public TorchYolo
	{
	public:
		TorchYoloPose() {}
		virtual ~TorchYoloPose() {}

		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
	};


	class TORCHAPP_API TorchYolo11Pose :public TorchYoloPose, virtual public TorchYolo11
	{
	public:
		TorchYolo11Pose() {}
		virtual ~TorchYolo11Pose() {}

		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	};


	class TORCHAPP_API TorchYolo26Pose :public TorchYoloPose, virtual  public TorchYolo26
	{
	public:
		TorchYolo26Pose() {}
		virtual ~TorchYolo26Pose() {}

		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	};


	class TORCHAPP_API TorchKeyPointRCNN :public TorchKeyPoint, virtual public TorchFasterRCNN
	{
	public:
		TorchKeyPointRCNN() {}
		virtual ~TorchKeyPointRCNN() {}

		virtual vector<DetectOutput> detectKeyPoint(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detectKeyPoint(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
	};

}

#endif