//#include "TorchRegress.h"
//#include "TorchClassify.h"
//#include "TorchCRNN.h"
//#include "TorchDetect.h"
//#include "TorchSegment.h"
//#include "TorchKeyPoint.h"
//#include "FileOperate.h"
//#include "AnnotationTransform.h"
//#include <Windows.h>
//
//using std::cout;
//using std::endl;
//using std::cin;
//using namespace torchapp;
//using namespace ljc;
//
////参数设定
//int top_N = 1;
//double scoreThreshold_detect = 0.25;
//double nmsThreshold_detect = 0.45;
//double scoreThreshold_segment = 0.25;	// 默认0.25 YoloV11-seg是0.45
//double nmsThreshold_segment = 0.45;	// 默认0.45 YoloV11-seg是0.5
//double maskThreshold_segment = 0.5;
//double scoreThreshold_keypoint = 0.5;
//static const int WIDTH = 300;
//static const int HEIGHT = 200;
//static const double OVERLAP = 0.2;
//
//typedef struct tagParams
//{
//	shared_ptr<TorchApp> pTorchApp;
//	string type;
//	cv::Size size;
//	bool useGPU;
//	int channel;
//	cv::Scalar mean;
//	cv::Scalar stdev;
//}Params, * pParams;
//
//Params paramYolo11Seg = {
//	std::make_shared<TorchYolo11Seg>(),
//	"yolo11seg",
//	{ 640,640 },
//	true,
//	3,
//	cv::Scalar::all(0),
//	cv::Scalar::all(1),
//};
//
//Params paramYolo26Seg = {
//	std::make_shared<TorchYolo11Seg>(),
//	"yolo26seg",
//	{ 640,640 },
//	true,
//	3,
//	cv::Scalar::all(0),
//	cv::Scalar::all(1),
//};
//
//Params paramMaskRCNN = {
//	std::make_shared<TorchMaskRCNN>(),
//	"maskRCNN",
//	cv::Size(),
//	true,
//	3,
//	cv::Scalar::all(0),
//	cv::Scalar::all(1),
//};
//
//vector<std::pair<Mat, cv::Rect>> Split(Mat img, int width, int height, double overlap)
//{
//	overlap = MIN(0.5, overlap);
//	int width_overlap = width * overlap + 0.5;
//	int height_overlap = height * overlap + 0.5;
//	vector<std::pair<Mat, cv::Rect>> results;
//	for (int h = 0; h < img.rows; h += height)
//		for (int w = 0; w < img.cols; w += width)
//		{
//			int x1 = w - width_overlap;
//			int x2 = w + width + width_overlap;
//			int y1 = h - height_overlap;
//			int y2 = h + height + height_overlap;
//			if (w + width >= img.cols)
//			{
//				x2 = img.cols - 1;
//				x1 = x2 - width - width_overlap;
//			}
//			if (h + height >= img.rows)
//			{
//				y2 = img.rows - 1;
//				y1 = y2 - height - height_overlap;
//			}
//			x1 = MAX(0, x1);
//			x2 = MIN(img.cols - 1, x2);
//			y1 = MAX(0, y1);
//			y2 = MIN(img.rows - 1, y2);
//			cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
//			results.push_back({ img(rect).clone(), rect });
//		}
//	return results;
//}
//
//void TestSegment(Params& params)
//{
//	std::shared_ptr<TorchSegment> pDetect = std::dynamic_pointer_cast<TorchSegment>(params.pTorchApp);
//	if (params.type == "maskRCNN")
//	{
//		std::dynamic_pointer_cast<TorchMaskRCNN>(pDetect)->setResize(false);
//		std::dynamic_pointer_cast<TorchMaskRCNN>(pDetect)->setMaskThresh(maskThreshold_segment);
//	}
//	else if (params.type == "yolo11seg")
//		std::dynamic_pointer_cast<TorchYolo11Seg>(pDetect)->setDetectParam(true, nmsThreshold_segment);
//	string model_file, label_file, path;
//	cout << "输入模型文件; ";
//	cin >> model_file;
//	cout << "输入标签文件; ";
//	cin >> label_file;
//	params.pTorchApp->initial(model_file, params.size, label_file, params.useGPU, params.channel, params.mean, params.stdev);
//	cout << "输入" + params.type + "图片文件夹: ";
//	cin >> path;
//	CreateDirectory((path + "_").c_str(), NULL);
//	auto files = FileOperate(path).getAllFiles(0);
//	vector<ImgInfo> vecImgInfo;
//	vector<cv::Scalar> colors;
//	srand(time(0));
//	for (int i = 0; i < pDetect->getClsNames().size(); i++)
//	{
//		int b = rand() % 256;
//		int g = rand() % 256;
//		int r = rand() % 256;
//		colors.push_back(cv::Scalar(b, g, r));
//	}
//	for (auto& file : files)
//	{
//		Mat img = cv::imread(path + "/" + file, -1);
//		if (!img.data)
//			continue;
//		Mat matMask = img.clone();
//		auto ImgRects = Split(img, img.cols / 3, img.rows / 2, img.rows / 10);
//		vector<Mat> imgs;
//		for (auto& imgrect : ImgRects)
//			imgs.push_back(imgrect.first);
//		auto results = pDetect->segment(imgs, scoreThreshold_segment);
//		for (int i = 0; i != results.size(); ++i)
//		{
//			for (auto& region : results[i])
//			{
//				cv::Rect bbox = region.box;
//				bbox.x += ImgRects[i].second.x;
//				bbox.y += ImgRects[i].second.y;
//				if (bbox.width >= 8 || bbox.height >= 8)
//					matMask(bbox).setTo(colors[region.cls_id], region.mask);
//			}
//		}
//		cv::addWeighted(img, 0.5, matMask, 0.5, 0, img);
//		cv::imwrite(path + "_/" + file, img);
//	}
//}
//
//int main()
//{
//	clock_t start0 = clock();
//	TestSegment(paramMaskRCNN);
//	double time0 = double(clock() - start0) / CLOCKS_PER_SEC;
//	cout << "用时" << time0 << "秒" << endl;
//	system("pause");
//}