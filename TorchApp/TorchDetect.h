#ifndef TORCHAPP_TORCHDETECT_H
#define TORCHAPP_TORCHDETECT_H

#include "TorchApp.h"

namespace at {
	class Tensor;
}

namespace torchapp {

	typedef struct tagDetectOutput
	{
		int cls_id;
		string cls_name;
		cv::Rect box;
		double score;
		Mat mask;
		vector<cv::Point> points;
	}DetectOutput, * pDetectOutput;


	class TORCHAPP_API TorchDetect :public TorchApp
	{
	public:
		TorchDetect() {}
		virtual ~TorchDetect() {}

		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25, vector<int> classes = {}) = 0;
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {}) = 0;

	};


	class TORCHAPP_API TorchYolo :virtual public TorchDetect
	{
	public:
		TorchYolo() :m_scale_fill(false), m_isCenter(true), m_padding_value(114) {}
		virtual ~TorchYolo() {}

		inline void setProcessParam(bool scale_fill = false, bool isCenter = true, int padding_value = 114)
		{
			m_scale_fill = scale_fill;
			m_isCenter = isCenter;
			m_padding_value = padding_value;
		}
		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	protected:
		Mat letterBox(Mat img);
		void scaleBox(cv::Rect& box, Mat img);
		Mat adjustMask(Mat mask, Mat origin_img, const cv::Rect& box);
		void scaleCoord(vector<cv::Point>& points, Mat img);
		// 对一张图的预测结果进行后处理, classes是需要考虑的类别id（为空表示考虑全部类别）
		virtual vector<DetectOutput> processOne(
			at::Tensor* pPredict,
			Mat origin_img,
			double scoreThreshold,
			vector<int> classes,
			at::Tensor* pProto = nullptr
		) = 0;

	protected:
		bool m_scale_fill;	// 是否直接拉伸图像至目标尺寸
		bool m_isCenter;	// 是否居中放置图像（false 则左上角对齐）
		int m_padding_value;	// 填充颜色值
	};


	class TORCHAPP_API TorchYolo11 :virtual public TorchYolo
	{
	public:
		TorchYolo11() :m_agnostic(true), m_nmsThreshold(0.45) {}
		virtual ~TorchYolo11() {}

		inline void setDetectParam(bool agnostic = true, double nmsThreshold = 0.45)
		{
			m_agnostic = agnostic;
			m_nmsThreshold = nmsThreshold;
		}
		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	protected:
		// 对一张图的预测结果进行后处理, classes是需要考虑的类别id（为空表示考虑全部类别）
		virtual vector<DetectOutput> processOne(
			at::Tensor *pPredict,
			Mat origin_img,
			double scoreThreshold,
			vector<int> classes,
			at::Tensor *pProto = nullptr
		);

	protected:
		bool m_agnostic;	// 为真表示类别无关nms
		double m_nmsThreshold;	// NMS阈值

	};


	class TORCHAPP_API TorchYolo26 :virtual public TorchYolo
	{
	public:
		TorchYolo26() {}
		virtual ~TorchYolo26() {}

		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25 , vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25 , vector<int> classes = {});

	protected:
		// 对批次图片的预测结果进行后处理, classes是需要考虑的类别id（为空表示考虑全部类别）
		virtual vector<DetectOutput> processOne(
			at::Tensor* pPredict,
			Mat origin_img,
			double scoreThreshold,
			vector<int> classes,
			at::Tensor* pProto = nullptr
		);
	};


	class TORCHAPP_API TorchFasterRCNN :virtual public TorchDetect
	{
	public:
		TorchFasterRCNN() : 
			m_bResize(true), 
			m_minSize(800), 
			m_maxSize(1333),
			m_nmsThreshold(0.45), 
			m_maskThreshold(0.5),
			m_scale_w(1.0),
			m_scale_h(1.0)
		{}
		virtual ~TorchFasterRCNN() {}

		inline void setResize(bool bResize, int min_size = 800, int max_size = 1333) 
		{ 
			m_bResize = bResize;
			m_minSize = min_size; 
			m_maxSize = max_size; 
		}
		inline void setThreshold(double nmsThreshold = 0.45, double maskThreshold = 0.5)
		{
			m_nmsThreshold = nmsThreshold;
			m_maskThreshold = maskThreshold;
		}
		virtual vector<DetectOutput> detect(Mat img, double scoreThreshold = 0.25, vector<int> classes = {});
		virtual vector<vector<DetectOutput>> detect(vector<Mat> imgs, double scoreThreshold = 0.25, vector<int> classes = {});

	protected:
		ModelOutput forward(Mat img);
		Mat resize(Mat img, int min_size = 800, int max_size = 1333);
		/* type :
		*		0 -- detect
		*		1 -- segment
		*		2 -- keypoint
		*/
		vector<DetectOutput> process(
			at::Tensor* pTensor_clsID,
			at::Tensor* pTensor_score,
			at::Tensor* pTensor_box,
			double scoreThreshold = 0.25, 
			vector<int> classes = {},
			at::Tensor* pTensor_mask = nullptr,
			at::Tensor* pTensor_keypoint = nullptr
		);

	protected:
		bool m_bResize;
		int m_minSize;
		int m_maxSize;
		double m_nmsThreshold;	// NMS阈值
		double m_maskThreshold;	// mask阈值
		double m_scale_w;	// 原图宽度/resize之后图片宽度
		double m_scale_h;	// 原图高度/resize之后图片高度
	};
}

#endif