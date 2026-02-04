#ifndef TORCHAPP_TORCHSEGMENT_H
#define TORCHAPP_TORCHSEGMENT_H

#include "TorchDetect.h"

namespace torchapp {

	class TORCHAPP_API TorchSegment :virtual public TorchDetect
	{
	public:
		TorchSegment() {}
		virtual ~TorchSegment() {}

		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {}) = 0;
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {}) = 0;
	};


	class TORCHAPP_API TorchYoloSeg :public TorchSegment, virtual public TorchYolo
	{
	public:
		TorchYoloSeg() { m_scale_fill = true; }	// 经实验开启m_scale_fill时效果最好
		virtual ~TorchYoloSeg() {}

		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	};


	class TORCHAPP_API TorchYolo11Seg :public TorchYoloSeg, virtual public TorchYolo11
	{
	public:
		TorchYolo11Seg() {}
		virtual ~TorchYolo11Seg() {}

		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
	};


	class TORCHAPP_API TorchYolo26Seg :public TorchYoloSeg, virtual public TorchYolo26
	{
	public:
		TorchYolo26Seg() {}
		virtual ~TorchYolo26Seg() {}

		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});
	};


	class TORCHAPP_API TorchMaskRCNN :public TorchSegment, virtual public TorchFasterRCNN
	{
	public:
		TorchMaskRCNN() :m_maskThreshold(0.5) {}
		virtual ~TorchMaskRCNN() {}

		inline void setMaskThresh(double maskThreshold) { m_maskThreshold = maskThreshold; }

		virtual vector<DetectOutput> segment(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> segment(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	private:
		double m_maskThreshold;

	};

}

#endif