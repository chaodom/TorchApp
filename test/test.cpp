#include "TorchFinal.h"
#include "FileOperate.h"
#include "AnnotationTransform.h"
#include <fstream>
#include <numeric>
#include <Windows.h>

using namespace std;
using namespace cv;
using namespace torchapp;
using namespace ljc;

vector<string> modes{ "regress", "classify", "crnn", "detect", "segment", "keypoint" };
vector<string> detect_types{ "yolo11", "yolo26", "rcnn" };
shared_ptr<TorchApp> pTorchApp;
vector<cv::Scalar> colors;
string path;
vector<Mat> imgs;
vector<string> img_files;
vector<ImgInfo> gt_infos;

static string UTF8ToGB(const char* str)
{
	string result;
	WCHAR* strSrc;
	LPSTR szRes;

	//获得临时变量的大小
	int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	strSrc = new WCHAR[i + 1];
	MultiByteToWideChar(CP_UTF8, 0, str, -1, strSrc, i);

	//获得临时变量的大小
	i = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
	szRes = new CHAR[i + 1];
	WideCharToMultiByte(CP_ACP, 0, strSrc, -1, szRes, i, NULL, NULL);

	result = szRes;
	delete[]strSrc;
	delete[]szRes;

	return result;
}

typedef struct tagParamsModel
{
	string model_file;
	string mode;	// regress, classify, crnn, detect, segment, keypoint
	cv::Size imgsz;
	bool useGPU;
	int channel;
	int batch_size;
	cv::Scalar mean;
	cv::Scalar stdev;
	string img_path;
}ParamsModel, * pParamsModel;
ParamsModel params;

typedef struct tagParamsClassify
{
	int topN;
	bool verify;
}ParamsClassify, * pParamsClassify;

typedef struct tagParamsDetect
{
	string detect_type;	// yolo11, yolo26, rcnn
	bool agnostic;	// NMS时是否类别无关
	double scoreThreshold;
	double nmsThreshold;
	vector<int> classes;
	//预处理参数
	bool scale_fill;// 是否直接拉伸图像至目标尺寸
	bool isCenter;// 是否居中放置图像（false 则左上角对齐）
	int padding_value;// 填充颜色值
	bool resize;	// rcnn系列预处理是否resize
	double epsilon;	// 简化轮廓时的多边形近似阈值
	bool verify;
	double iou;
}ParamsDetect, * pParamsDetect;

ParamsModel LoadParamsModel(const string& file = path + "/settings_model.xml")
{
	ParamsModel params;
	params.model_file = "best.torchscript";
	params.imgsz = cv::Size();
	params.useGPU = true;
	params.channel = 3;
	params.batch_size = 1;
	params.mean = cv::Scalar::all(0);
	params.stdev = cv::Scalar::all(1.0);
	params.img_path = "iamges";
	cv::FileStorage fin(file, cv::FileStorage::READ);
	if (fin.isOpened())
	{
		cv::FileNode node = fin["ModelFile"];
		if (!node.empty())
			node >> params.model_file;
		fin["Mode"] >> params.mode;
		fin["Imgsz"] >> params.imgsz;
		node = fin["UseGPU"];
		if (!node.empty())
			node >> params.useGPU;
		node = fin["Channel"];
		if (!node.empty())
			node >> params.channel;
		node = fin["BatchSize"];
		if (!node.empty())
			node >> params.batch_size;
		fin["Mean"] >> params.mean;
		node = fin["Stdev"];
		if (!node.empty())
			node >> params.stdev;
		node = fin["ImagePath"];
		if (!node.empty())
			node >> params.img_path;
		params.img_path = UTF8ToGB(params.img_path.c_str());
		fin.release();
	}
	return params;
}

ParamsClassify LoadParamsClassify(const string& file = path + "/settings_classify.xml")
{
	ParamsClassify params;
	params.topN = 1;
	params.verify = 0;
	cv::FileStorage fin(file, cv::FileStorage::READ);
	if (fin.isOpened())
	{
		cv::FileNode node = fin["TopN"];
		if (!node.empty())
			node >> params.topN;
		fin["Verify"] >> params.verify;
		fin.release();
	}
	return params;
}

ParamsDetect LoadParamsDetect(const string& file = path + "/settings_detect.xml")
{
	ParamsDetect params;
	params.agnostic = true;
	params.scoreThreshold = 0.25;
	params.nmsThreshold = 0.45;
	params.classes = {};
	params.scale_fill = false;
	params.isCenter = true;
	params.resize = true;
	params.padding_value = 114;
	params.epsilon = 0.001;
	params.verify = 0;
	params.iou = 0.9;
	cv::FileStorage fin(file, cv::FileStorage::READ);
	if (fin.isOpened())
	{
		fin["DetectType"] >> params.detect_type;
		cv::FileNode node = fin["Agnostic"];
		if (!node.empty())
			node >> params.agnostic;
		node = fin["ScoreThreshold"];
		if (!node.empty())
			node >> params.scoreThreshold;
		node = fin["NMSThreshold"];
		if (!node.empty())
			node >> params.nmsThreshold;
		fin["IncludeClasses"] >> params.classes;
		fin["ScaleFill"] >> params.scale_fill;
		node = fin["IsCenter"];
		if (!node.empty())
			node >> params.isCenter;
		node = fin["PaddingValue"];
		if (!node.empty())
			node >> params.padding_value;
		node = fin["Resize"];
		if (!node.empty())
			node >> params.resize;
		node = fin["Epsilon"];
		if (!node.empty())
			node >> params.epsilon;
		fin["Verify"] >> params.verify;
		node = fin["IOU"];
		if (!node.empty())
			node >> params.iou;
		fin.release();
	}
	return params;
}

void LoadImages(const string& path)
{
	imgs.clear();
	img_files.clear();
	auto files = FileOperate(path).getAllFiles(false);
	Mat img;
	for (auto& file : files)
	{
		if (params.channel == 1)
			img = imread(path + "/" + file, 0);
		else if (params.channel == 3)
			img = imread(path + "/" + file, 1);
		else
			img = imread(path + "/" + file, -1);
		if (img.data)
		{
			imgs.push_back(img);
			img_files.push_back(file);
		}
	}
}

void TestRegress(ParamsModel& p)
{
	if (imgs.empty())
	{
		cerr << p.img_path + "内无图片" << endl;
		return;
	}
	pTorchApp = make_shared<TorchRegress>();
	pTorchApp->initial(path + "/" + p.model_file, p.imgsz, path + "/labels.txt", p.useGPU, p.channel, p.mean, p.stdev);
	shared_ptr<TorchRegress> pRegress = dynamic_pointer_cast<TorchRegress>(pTorchApp);
	// 补充图片数量为batch_size的整倍数
	int num = imgs.size() % params.batch_size ? (imgs.size() / params.batch_size + 1) * params.batch_size : imgs.size();
	Mat empty_img = Mat::zeros(imgs[0].size(), imgs[0].type());
	imgs.resize(num, empty_img);
	vector<vector<float>> outputs;
	outputs.reserve(imgs.size());
	auto it = imgs.begin();
	// 开始检测
	clock_t start0 = clock();
	while (it != imgs.end())
	{
		vector<Mat> batch_imgs(it, it + params.batch_size);
		auto results = pRegress->regress(batch_imgs);
		outputs.insert(outputs.end(), results.begin(), results.end());
		it += params.batch_size;
		Sleep(1);
	}
	double time0 = double(clock() - start0) / CLOCKS_PER_SEC;
	cout << "用时" << time0 << "秒" << endl;
	imgs.resize(img_files.size());
	for (int i = 0; i != imgs.size(); ++i)
	{
		cout << img_files[i] << "\t";
		for (auto& res : outputs[i])
			cout << res << "\t";
		cout << endl;
	}
}

void ClassifyImages(ParamsClassify& p, vector<vector<ClassifyOutput>>& outputs)
{
	shared_ptr<TorchClassify> pClassify = dynamic_pointer_cast<TorchClassify>(pTorchApp);
	// 补充图片数量为batch_size的整倍数
	int num = imgs.size() % params.batch_size ? (imgs.size() / params.batch_size + 1) * params.batch_size : imgs.size();
	Mat empty_img = Mat::zeros(imgs[0].size(), imgs[0].type());
	imgs.resize(num, empty_img);
	outputs.reserve(imgs.size());
	auto it = imgs.begin();
	while (imgs.end() - it >= params.batch_size)
	{
		vector<Mat> batch_imgs(it, it + params.batch_size);
		vector<vector<ClassifyOutput>> results = pClassify->classify(batch_imgs, p.topN);
		outputs.insert(outputs.end(), results.begin(), results.end());
		it += params.batch_size;
		Sleep(1);
	}
	imgs.resize(img_files.size());
}

void ClassifyPath(const string& img_path, void* pDataP)
{
	ParamsClassify pc = *static_cast<ParamsClassify*>(pDataP);
	LoadImages(img_path);
	if (imgs.empty())
		return;
	vector<vector<ClassifyOutput>> outputs;
	ClassifyImages(pc, outputs);
	string gt_cls = img_path.substr(img_path.find_last_of("/\\") + 1);
	int correct = 0;
	for (int i = 0; i != imgs.size(); ++i)
	{
		bool found = false;
		for (auto& res : outputs[i])
			if (res.cls_name == gt_cls)
			{
				++correct;
				found = true;
				break;
			}
		if (!found)
		{
			CreateDirectory("false", NULL);
			CreateDirectory(("false\\" + gt_cls).c_str(), NULL);
			CopyFile((img_path + "\\" + img_files[i]).c_str(), ("false\\" + gt_cls + "\\" + img_files[i]).c_str(), TRUE);
		}
	}
	cout << gt_cls << "\t正确率\t" << static_cast<double>(correct) / imgs.size() << "\t总数\t" << imgs.size() << endl;
}

void TestClassify(ParamsModel& p)
{
	ParamsClassify pc = LoadParamsClassify();
	pTorchApp = make_shared<TorchClassify>();
	pTorchApp->initial(path + "/" + p.model_file, p.imgsz, path + " / labels.txt", p.useGPU, p.channel, p.mean, p.stdev);
	// 开始运算
	clock_t start0 = clock();
	vector<vector<ClassifyOutput>> outputs;
	if (pc.verify)
		FileOperate(path).proEachFile(nullptr, nullptr, ClassifyPath, &pc);
	else
	{
		if (imgs.empty())
		{
			cerr << p.img_path + "内无图片" << endl;
			return;
		}
		ClassifyImages(pc, outputs);
		for (int i = 0; i != imgs.size(); ++i)
		{
			cout << img_files[i] << "\t";
			for (auto& res : outputs[i])
				cout << res.cls_name << "(" << res.score << ")\t";
			cout << endl;
		}
	}
	double time0 = double(clock() - start0) / CLOCKS_PER_SEC;
	cout << "用时" << time0 << "秒" << endl;
}

void TestCRNN(ParamsModel& p)
{
	if (imgs.empty())
	{
		cerr << p.img_path + "内无图片" << endl;
		return;
	}
	pTorchApp = make_shared<TorchCRNN>();
	pTorchApp->initial(path + "/" + p.model_file, p.imgsz, path + "/labels.txt", p.useGPU, p.channel, p.mean, p.stdev);
	shared_ptr<TorchCRNN> pCRNN = dynamic_pointer_cast<TorchCRNN>(pTorchApp);
	clock_t start0 = clock();
	for (int i = 0; i != imgs.size(); ++i)
	{
		auto result = pCRNN->recognize(imgs[i]);
		cout << img_files[i] << "\t";
		cout << result.text << "( ";
		for (auto& score : result.scores)
			cout << score << " ";
		cout << ")" << endl;
	}
	double time0 = double(clock() - start0) / CLOCKS_PER_SEC;
	cout << "用时" << time0 << "秒" << endl;
}

double ContentIOU(const vector<int>& content1, const vector<int>& content2)
{
	double areaInter = 0.0;
	double areaUnion = 0.0;
	if (params.mode == "detect" || params.mode == "keypoint")
	{
		// 计算矩形交集
		int dx = MIN(content1[0] + content1[2], content2[0] + content2[2]) - MAX(content1[0], content2[0]);
		int dy = MIN(content1[1] + content1[3], content2[1] + content2[3]) - MAX(content1[1], content2[1]);
		areaInter = MAX(0, dx) * MAX(0, dy);
		// 计算矩形并集
		int area1 = content1[2] * content1[3];
		int area2 = content2[2] * content2[3];
		areaUnion = area1 + area2 - areaInter;
	}
	else if (params.mode == "segment")
	{
		vector<cv::Point> contour1, contour2;
		contour1.reserve(content1.size() / 2);
		contour2.reserve(content2.size() / 2);
		for (int i = 0; i != content1.size(); i += 2)
			contour1.push_back(cv::Point(content1[i], content1[i + 1]));
		for (int i = 0; i != content2.size(); i += 2)
			contour2.push_back(cv::Point(content2[i], content2[i + 1]));
		Rect rect1 = boundingRect(contour1);
		Rect rect2 = boundingRect(contour2);
		// 联合矩形：覆盖两个轮廓的最小矩形（仅在该区域生成掩码，减少无效运算）
		Rect unionRect(
			min(rect1.x, rect2.x),    // x1
			min(rect1.y, rect2.y),    // y1
			max(rect1.x + rect1.width, rect2.x + rect2.width) - min(rect1.x, rect2.x),  // width
			max(rect1.y + rect1.height, rect2.y + rect2.height) - min(rect1.y, rect2.y) // height
		);
		// 生成两个轮廓在联合矩形内的二值掩码
		Mat mask1 = Mat::zeros(unionRect.size(), CV_8U);
		Mat mask2 = Mat::zeros(unionRect.size(), CV_8U);
		// 轮廓坐标偏移：将轮廓点转换为联合矩形内的相对坐标（避免超出掩码范围）
		vector<Point> contour1_offset, contour2_offset;
		for (const auto& p : contour1)
			contour1_offset.emplace_back(p.x - unionRect.x, p.y - unionRect.y);
		for (const auto& p : contour2)
			contour2_offset.emplace_back(p.x - unionRect.x, p.y - unionRect.y);
		// 填充轮廓为白色
		fillPoly(mask1, vector<vector<Point>>{contour1_offset}, Scalar(255));
		fillPoly(mask2, vector<vector<Point>>{contour2_offset}, Scalar(255));
		// 计算面积
		int area1 = countNonZero(mask1);
		int area2 = countNonZero(mask2);
		if (area1 == 0 || area2 == 0)
			return 0.0;
		// 计算交集面积（bitwise_and 是逐像素与运算，底层优化）
		Mat maskIntersection;
		bitwise_and(mask1, mask2, maskIntersection);
		areaInter = countNonZero(maskIntersection);
		areaUnion = area1 + area2 - areaInter;
	}
	return areaUnion > 0 ? areaInter / areaUnion : 0.0;
}

bool VerifyInfo(const ImgInfo& info1, const ImgInfo& info2)
{
	if (info1.objs.size() != info2.objs.size())
		return false;
	vector<bool> matched(info2.objs.size(), false);	// 记录info2中每个目标是否已被匹配
	for (const auto& obj1 : info1.objs)
	{
		bool found = false;
		for (size_t j = 0; j < info2.objs.size(); ++j)
		{
			// 只匹配未被使用的info2目标
			if (!matched[j] && ContentIOU(obj1.content, info2.objs[j].content) >= 0.5)
			{
				found = true;
				matched[j] = true; // 标记为已匹配
				break;
			}
		}
		if (!found)
			return false;
	}
	return true;
}

void ShowImg(Mat img, const string& save_path, const string& img_file, vector<DetectOutput> results, ParamsDetect& pd)
{
	ImgInfo img_info;
	img_info.file = img_file;
	img_info.height = img.rows;
	img_info.width = img.cols;
	img_info.objs.reserve(results.size());
	Mat matMask = img.clone();
	for (auto& res : results)
	{
		cv::Scalar color = res.cls_id < colors.size() ? colors[res.cls_id] : cv::Scalar(114, 114, 114);
		int left = res.box.x;
		int top = res.box.y;
		rectangle(img, res.box, color, 2);
		string label = res.cls_name + ":" + to_string(res.score);
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = MAX(top, labelSize.height);
		cv::putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
		ObjInfo obj;
		obj.cls_id = res.cls_id;
		obj.cls_name = res.cls_name;
		obj.diff = 0;
		if (params.mode == "segment")
		{
			matMask(res.box).setTo(color, res.mask);
			vector<vector<cv::Point>> contours; // 用于存储所有轮廓的向量
			vector<cv::Vec4i> hierarchy; // 用于存储轮廓的层次关系（可选）
			cv::findContours(res.mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // 查找轮廓
			int index = 0;
			double max_area = 0;
			for (int i = 0; i != contours.size(); ++i)
			{
				double area = cv::contourArea(contours[i]);
				if (area > max_area)
				{
					index = i;
					max_area = area;
				}
			}
			vector<cv::Point> points;
			double epsilon = pd.epsilon * arcLength(contours[index], true); // 简化阈值（基于轮廓周长）
			approxPolyDP(contours[index], points, epsilon, true);            // 多边形逼近
			for (auto& p : points)
			{
				obj.content.push_back(p.x + res.box.x);
				obj.content.push_back(p.y + res.box.y);
			}
		}
		else if (params.mode == "keypoint")
		{
			obj.content = { res.box.x, res.box.y, res.box.width, res.box.height };
			for (auto& point : res.points)
			{
				cv::circle(img, point, 2, color, -1);
				obj.content.push_back(point.x);
				obj.content.push_back(point.y);
				obj.content.push_back(2);
			}
		}
		else
			obj.content = { res.box.x, res.box.y, res.box.width, res.box.height };
		img_info.objs.push_back(obj);
	}
	cv::addWeighted(img, 0.5, matMask, 0.5, 0, img);
	AddAnnotation2File(img_info, save_path + "/annotation.txt");
	if (pd.verify)
	{
		auto it = find_if(gt_infos.begin(), gt_infos.end(), [img_file](const ImgInfo& info) {return info.file == img_file; });
		if (it == gt_infos.end())
		{
			CreateDirectory((save_path + "nolabel").c_str(), NULL);
			imwrite(save_path + "nolabel/" + img_file, img);
		}
		else
		{
			auto standard_info = *it;
			if (!VerifyInfo(img_info, standard_info))
			{
				CreateDirectory((save_path + "wrong").c_str(), NULL);
				AddAnnotation2File(img_info, path + "_wrong/annotation.txt");
				imwrite(save_path + "wrong/" + img_file, img);
			}
		}
	}
	imwrite(save_path + "/" + img_file, img);
}

void TestDetect(ParamsModel& p)
{
	if (imgs.empty())
	{
		cerr << p.img_path + "内无图片" << endl;
		return;
	}
	ParamsDetect pd = LoadParamsDetect();
	// 构建实例
	if (pd.detect_type.substr(0, 4) == "yolo")
	{
		if (pd.detect_type == "yolo11")	// yolo11
		{
			pTorchApp = make_shared<TorchYolo11Final>();
			dynamic_pointer_cast<TorchYolo11>(pTorchApp)->setDetectParam(pd.agnostic, pd.nmsThreshold);
		}
		else if (pd.detect_type == "yolo26")	// yolo26
			pTorchApp = make_shared<TorchYolo26Final>();
		dynamic_pointer_cast<TorchYolo>(pTorchApp)->setProcessParam(pd.scale_fill, pd.isCenter, pd.padding_value);
	}
	else if (pd.detect_type == "rcnn")
	{
		pTorchApp = make_shared<TorchRCNNFinal>();
		dynamic_pointer_cast<TorchFasterRCNN>(pTorchApp)->setResize(pd.resize);
		dynamic_pointer_cast<TorchFasterRCNN>(pTorchApp)->setThreshold(pd.nmsThreshold);
	}
	else
	{
		cerr << "detect_type必须是以下之一:[ ";
		for (auto& type : detect_types)
			cerr << type << " ";
		cerr << "]" << endl;
		return;
	}
	// 模型初始化
	pTorchApp->initial(path + "/" + p.model_file, p.imgsz, path + "/labels.txt", p.useGPU, p.channel, p.mean, p.stdev);
	if (pd.verify)
		gt_infos = ReadAnnotation(path + "/" + p.img_path + "/annotation.txt");
	// 补充图片数量为batch_size的整倍数
	int num = imgs.size() % params.batch_size ? (imgs.size() / params.batch_size + 1) * params.batch_size : imgs.size();
	Mat empty_img = Mat::zeros(imgs[0].size(), imgs[0].type());
	imgs.resize(num, empty_img);
	vector<vector<DetectOutput>> outputs;
	outputs.reserve(imgs.size());
	auto it = imgs.begin();
	// 开始检测
	clock_t start0 = clock();
	while (it != imgs.end())
	{
		vector<Mat> batch_imgs(it, it + params.batch_size);
		vector<vector<DetectOutput>> results;
		if (p.mode == "detect")	//检测
			results = dynamic_pointer_cast<TorchDetect>(pTorchApp)->detect(batch_imgs, pd.scoreThreshold, pd.classes);
		else if (p.mode == "segment")	//分割
			results = dynamic_pointer_cast<TorchSegment>(pTorchApp)->segment(batch_imgs, pd.scoreThreshold, pd.classes);
		else if (p.mode == "keypoint")
			results = dynamic_pointer_cast<TorchKeyPoint>(pTorchApp)->detectKeyPoint(batch_imgs, pd.scoreThreshold, pd.classes);
		outputs.insert(outputs.end(), results.begin(), results.end());
		it += params.batch_size;
		Sleep(1);
	}
	double time0 = double(clock() - start0) / CLOCKS_PER_SEC;
	cout << "用时" << time0 << "秒" << endl;
	imgs.resize(img_files.size());
	string save_path = path + "/" + p.img_path + "_";
	CreateDirectory(save_path.c_str(), NULL);
	for (int i = 0; i != imgs.size(); ++i)
	{
		Mat img = imgs[i];
		ShowImg(imgs[i], save_path, img_files[i], outputs[i], pd);
	}
}

int main()
{
	std::ifstream fin("colors.txt");
	string line;
	while (std::getline(fin, line))
	{
		std::istringstream sin(line);
		int b, g, r;
		sin >> b >> g >> r;
		colors.push_back(cv::Scalar(b, g, r));
	}
	fin.close();
	cerr << "输入文件夹,文件夹内包含\n模型文件--best.torchscript\n标签文件--labels.txt（可选）\n模型配置文件--settings_model.xml\n待识别图片文件夹--images\n：";
	cin >> path;
	params = LoadParamsModel();
	LoadImages(path + "/" + params.img_path);
	if (params.mode == "regress")
		TestRegress(params);
	else if (params.mode == "classify")
		TestClassify(params);
	else if (params.mode == "crnn")
		TestCRNN(params);
	else if (params.mode == "detect" || params.mode == "segment" || params.mode == "keypoint")
		TestDetect(params);
	else
	{
		cerr << "mode必须是以下之一:[ ";
		for (auto& mode : modes)
			cerr << mode << " ";
		cerr << "]" << endl;
	}
	system("pause");
}