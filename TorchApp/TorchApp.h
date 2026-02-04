#ifndef TORCHAPP_TORCHAPP_H
#define TORCHAPP_TORCHAPP_H

#include <opencv2/opencv.hpp>

#ifdef TORCHAPP_DLLEXPORT
#define TORCHAPP_API __declspec(dllexport)
#else
#define TORCHAPP_API __declspec(dllimport)
#endif

using std::string;
using std::vector;
using std::shared_ptr;
using cv::Mat;

namespace torch {
	namespace jit {
		class Module;
	}
}
namespace c10 {
	extern struct IValue;
}

namespace torchapp {

	class TORCHAPP_API TorchApp
	{
	public:
		typedef torch::jit::Module Model;
		typedef c10::IValue ModelOutput;

		TorchApp() {}
		virtual ~TorchApp() {}

		inline vector<string> getClsNames() { return m_vecClsNames; }
		void initial(const string& model_file, cv::Size size = cv::Size(), const string& label_file = "", bool useGPU = false, int channel = 3, cv::Scalar mean = cv::Scalar::all(0), cv::Scalar stdev = cv::Scalar::all(1), const string& key = "");

	protected:
		ModelOutput forward(Mat img, bool swapRB = false);
		ModelOutput forward(vector<Mat> imgs, bool swapRB = false);

	protected:
		shared_ptr<Model> m_pModel;
		int m_iChannel;
		cv::Size m_size;
		cv::Scalar m_mean;	// 归一化后均值
		cv::Scalar m_stdev;
		bool m_bUseGPU;
		vector<string> m_vecClsNames;
	};

}

#endif