#include "TorchFinal.h"

namespace torchapp {

	vector<DetectOutput> TorchYolo11Final::detect(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo11::detect(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo11Final::detect(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo11::detect(imgs, scoreThreshold, classes);
	}

	vector<DetectOutput> TorchYolo11Final::segment(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo11Seg::segment(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo11Final::segment(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo11Seg::segment(imgs, scoreThreshold, classes);
	}

	vector<DetectOutput> TorchYolo11Final::detectKeyPoint(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo11Pose::detectKeyPoint(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo11Final::detectKeyPoint(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo11Pose::detectKeyPoint(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchYolo26Final::detect(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo26::detect(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo26Final::detect(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo26::detect(imgs, scoreThreshold, classes);
	}

	vector<DetectOutput> TorchYolo26Final::segment(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo26Seg::segment(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo26Final::segment(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo26Seg::segment(imgs, scoreThreshold, classes);
	}

	vector<DetectOutput> TorchYolo26Final::detectKeyPoint(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo26Pose::detectKeyPoint(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo26Final::detectKeyPoint(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo26Pose::detectKeyPoint(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchRCNNFinal::detect(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchFasterRCNN::detect(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchRCNNFinal::detect(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchFasterRCNN::detect(imgs, scoreThreshold, classes);
	}

	vector<DetectOutput> TorchRCNNFinal::segment(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchMaskRCNN::segment(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchRCNNFinal::segment(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchMaskRCNN::segment(imgs, scoreThreshold, classes);
	}

	vector<DetectOutput> TorchRCNNFinal::detectKeyPoint(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchKeyPointRCNN::detectKeyPoint(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchRCNNFinal::detectKeyPoint(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchKeyPointRCNN::detectKeyPoint(imgs, scoreThreshold, classes);
	}

}